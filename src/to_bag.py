#!/usr/bin/env python3
"""
Converter from dataset folders to ROS 2 bags.

This script builds a bag from raw LiDAR files plus CSV
IMU/ground-truth tables.
It shares the following config with rosbag1_to_ros2.py:
``output:`` keys, ``tf_static:`` schema, per-topic knobs (``frame_id``,
``min_dt_sec``, ``write_stamp_shift_sec``), the ``time_field``/``time_unit``/
``ring_field`` vocabulary, and the named cloud layouts in ``pc2_layout`` -- so a
bag built either way is interchangeable downstream.

Config shape (see bag generation in README.md)::

    output:   {storage_id, version}
    sources:  [ {out, type, frame_id, ...}, ... ]   # one entry per output topic
    tf_static: [ {parent, child, translation|matrix, ...}, ... ]

Adding a dataset should mean writing YAML, not code. The two halves that used to
be hardcoded are now config:

  * **input layout** -- each reader emits the canonical per-point names in
    ``pc2_layout.CANONICAL_POINT_FIELDS``; source field names and units are
    config (``time_field``, ``time_unit``, ``ring_field``, ``intensity_field``).
  * **CSV column order** -- ``columns:`` names each column positionally, so an
    11-column ConPR IMU and a 7-column EuRoC IMU are both just a list.

Runs on pure-python ``rosbags``, no ROS environment needs to be sourced.
"""
import argparse
import glob
import os
from typing import Any

import numpy as np

import resource_log
from pc2_layout import (
    CANONICAL_POINT_FIELDS, DEFAULT_INTENSITY_FIELDS, DEFAULT_RING_FIELDS,
    DEFAULT_TIME_FIELDS, NAME_TO_DATATYPE, PC2_DATATYPES, SCAN_SPAN_MAX_S,
    TIME_UNIT_TO_NS, TIME_UNIT_TO_S,
    build_layout, epoch_kind, normalize_layout, reset_warnings, span_warning,
    to_seconds, warn_once,
)
from pcd_io import parse_pcd
from ros2_bag_io import (
    ROS2_STORE, add_distro_arg, apply_distro, atomic_bag, check_window, load_yaml,
    ns_to_time, normalize_topic, output_summary, write_tf_static,
)

SOURCE_TYPES = ("pointcloud2", "imu", "odometry")
# Point-cloud sources: read per-frame files rather than a CSV table. Kept as a
# tuple (rather than folded into `== "pointcloud2"`) because prepare(), convert()
# and batch_to_bag.required_inputs() all branch on "is this a per-frame source",
# which is a different question from "which message type".
CLOUD_TYPES = ("pointcloud2",)

# Canonical CSV column names understood by `columns:`. "_" skips a column.
IMU_COLUMNS = ("t", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az")
ODOM_COLUMNS = ("t", "x", "y", "z", "qx", "qy", "qz", "qw")

# {topic: {"frames": n, "points": n}} for `time_outliers: drop`, reported in the
# run summary so the scale of a per-frame data defect is visible, not just its
# existence (warn_once only ever prints one line per topic).
OUTLIER_TALLY: dict[str, dict[str, int]] = {}

INT64_MIN, INT64_MAX = int(np.iinfo(np.int64).min), int(np.iinfo(np.int64).max)

# ConPR's Livox .bin: the raw CustomPoint struct dumped back to back.
LIVOX_BIN_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("refl", "u1"), ("tag", "u1"), ("line", "u1"),
    ("offset", "<u4"),
])


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------
def resolve_path(seq_dir: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(seq_dir, value)


def stamp_scale(unit: str) -> float:
    if unit not in TIME_UNIT_TO_NS:
        raise SystemExit(f"unknown stamp_unit {unit!r}; expected one of {sorted(TIME_UNIT_TO_NS)}")
    return TIME_UNIT_TO_NS[unit]


def stamps_to_ns(values: np.ndarray, unit: str, shift_ns: int, context: str = "") -> np.ndarray:
    """Whole-message timestamps -> int64 nanoseconds.

    A Unix-nanosecond stamp (~1.7e18) is far past float64's 2**53 exact-integer
    range -- one ULP up there is 256 ns -- so an integer input is carried through
    as int64 rather than being routed through a float multiply. Callers must hand
    us an integer array to get that; see read_int_column / list_frames.

    Every path is range-checked, because the unchecked casts fail *silently*:
    ``np.rint(nan).astype(np.int64)`` and ``1e30`` both land on INT64_MIN, and a
    uint64 above 2**63 wraps negative -- each of which yields a bag whose clock is
    nonsense rather than an error.
    """
    values = np.asarray(values)
    if values.dtype.kind == "u" and values.size and int(values.max()) > INT64_MAX:
        raise SystemExit(f"{_label(context)}timestamp {int(values.max())} exceeds the int64 "
                         f"nanosecond range; the source cannot be a Unix stamp")
    if values.dtype.kind in "iu" and unit == "ns":
        ns = values.astype(np.int64)   # exact: no float round trip
    else:
        scaled = values.astype(np.float64)
        if unit != "ns":
            scaled = scaled * stamp_scale(unit)
        check_finite(scaled, context, "timestamp")
        _check_int64_range(scaled, context, unit)
        ns = np.rint(scaled).astype(np.int64)
    if shift_ns and ns.size:
        lo, hi = int(ns.min()) + shift_ns, int(ns.max()) + shift_ns
        if lo < INT64_MIN or hi > INT64_MAX:
            raise SystemExit(f"{_label(context)}applying a {shift_ns} ns stamp shift takes the "
                             f"timestamps outside the int64 nanosecond range")
    return ns + np.int64(shift_ns)


def _label(context: str) -> str:
    return f"[{context}] " if context else ""


def check_finite(values: np.ndarray, context: str, what: str) -> None:
    """Reject NaN/Inf. Fatal for timestamps: they define the bag's clock, and a
    non-finite one casts to INT64_MIN without complaint."""
    bad = ~np.isfinite(values)
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        raise SystemExit(f"{_label(context)}{int(bad.sum())} of {bad.size} {what}(s) are NaN/Inf "
                         f"(first at index {first}) -- fix the source data; a non-finite "
                         f"timestamp cannot be written to a bag")


def _check_int64_range(scaled: np.ndarray, context: str, unit: str) -> None:
    if not scaled.size:
        return
    lo, hi = float(scaled.min()), float(scaled.max())
    if lo < INT64_MIN or hi > INT64_MAX:
        raise SystemExit(f"{_label(context)}timestamps span [{lo:.6g}, {hi:.6g}] ns when read as "
                         f"stamp_unit: {unit}, outside the int64 nanosecond range -- check "
                         f"`stamp_unit` (a seconds column read as ns, or vice versa, is off by 1e9)")


def read_int_column(path: str, delimiter: str | None, skiprows: int, col: int) -> np.ndarray | None:
    """Re-read one column as int64, or None if it is not integral.

    Only used for timestamp columns: np.loadtxt's float64 default silently
    rounds ns Unix stamps to the nearest ~256 ns.
    """
    try:
        return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows,
                          usecols=(col,), dtype=np.int64, ndmin=1)
    except ValueError:
        return None


def shift_ns_of(cfg: dict[str, Any]) -> int:
    if "stamp_shift_ns" in cfg:
        return int(cfg["stamp_shift_ns"])
    return int(round(float(cfg.get("stamp_shift_sec", 0.0)) * 1e9))


def check_epoch(stamps_ns: np.ndarray, topic: str, what: str) -> None:
    """Flag stamps that are not plausible Unix time.

    Catches the classic trap of index-named frames (`000001.pcd` parsed as 1 ns)
    and of a wrong ``stamp_unit``, both of which otherwise produce a bag whose
    clock is silently nonsense.
    """
    if stamps_ns.size == 0:
        return
    kind = epoch_kind(int(stamps_ns[0]))
    if kind != "unix_absolute":
        warn_once(f"{topic}:{what}", (
            f"{what} stamps look {kind} (first={int(stamps_ns[0])} ns) rather than "
            f"absolute Unix time -- check `stamp_unit`, or use `stamp_from: file` "
            f"if the filenames are frame indices rather than timestamps"
        ))


# --------------------------------------------------------------------------
# Frame listing / per-frame timestamps
# --------------------------------------------------------------------------
def list_frames(seq_dir: str, src: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Sorted (paths, stamps_ns) for one per-frame LiDAR source."""
    data_dir = resolve_path(seq_dir, src["dir"])
    if not os.path.isdir(data_dir):
        raise SystemExit(f"[{src['out']}] lidar dir not found: {data_dir}")
    paths = sorted(glob.glob(os.path.join(data_dir, src.get("pattern", "*"))))
    if not paths:
        raise SystemExit(f"[{src['out']}] no files matching "
                         f"{src.get('pattern', '*')!r} in {data_dir}")

    unit = src.get("stamp_unit", "ns")
    shift = shift_ns_of(src)
    stamp_from = src.get("stamp_from", "filename")
    if stamp_from == "filename":
        raw, all_int = [], True
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            try:                       # int first: a 19-digit ns stamp does not
                raw.append(int(stem))  # survive a float64 round trip
            except ValueError:
                all_int = False
                try:
                    raw.append(float(stem))
                except ValueError:
                    raise SystemExit(
                        f"[{src['out']}] filename {os.path.basename(p)!r} is not a number, so it "
                        f"carries no timestamp; point `stamp_file:` at the dataset's timestamp "
                        f"list and set `stamp_from: file`"
                    ) from None
        stamps = stamps_to_ns(np.array(raw, dtype=np.int64 if all_int else np.float64),
                              unit, shift, src["out"])
    elif stamp_from == "file":
        if "stamp_file" not in src:
            raise SystemExit(f"[{src['out']}] stamp_from: file requires `stamp_file:`")
        stamp_path = resolve_path(seq_dir, src["stamp_file"])
        delim = src.get("stamp_delimiter")
        col = int(src.get("stamp_column", 0))
        column = read_int_column(stamp_path, delim, 0, col)
        if column is None:
            column = np.loadtxt(stamp_path, ndmin=2, delimiter=delim)[:, col]
        if len(column) != len(paths):
            raise SystemExit(
                f"[{src['out']}] stamp_file has {len(column)} rows but {len(paths)} files "
                f"match {src.get('pattern', '*')!r} -- they must correspond one to one"
            )
        stamps = stamps_to_ns(column, unit, shift, src["out"])
    else:
        raise SystemExit(f"[{src['out']}] unknown stamp_from {stamp_from!r}; use filename|file")

    check_epoch(stamps, src["out"], "frame")
    order = np.argsort(stamps, kind="stable")   # filenames need not sort numerically
    return [paths[i] for i in order], stamps[order]


# --------------------------------------------------------------------------
# Point-cloud readers -> canonical per-point columns
#
# Every reader returns {canonical name: 1-D array}. `time` is always SECONDS
# relative to the frame stamp. Missing keys are fine -- the layout mapper zero
# fills and warns.
# --------------------------------------------------------------------------
def pick_field(available: Any, configured: str | None, defaults: tuple[str, ...]) -> str | None:
    """Resolve a source field name: explicit config wins, else first known alias."""
    if configured is not None:
        return configured if configured in available else None
    for name in defaults:
        if name in available:
            return name
    return None


def relative_seconds(raw: np.ndarray, unit: str, stamp_ns: int, src: dict[str, Any],
                     field_name: str, path: str = "") -> np.ndarray:
    """Per-point time column -> seconds relative to the frame stamp.

    Handles both conventions seen in the wild: scan-relative offsets (Livox,
    Velodyne) pass through scaled, absolute Unix time (Hesai) is differenced
    against the frame stamp -- the same choice ``repack_hesai_to_velodyne`` makes
    for ROS 1 input. The observed span is checked so a wrong ``time_unit``, which
    would otherwise scale silently by a power of 1000, gets a warning.

    ``time_outliers: signed`` additionally recovers an UNSIGNED per-point time
    that underflowed. A driver computing ``point_time - scan_start`` in a uint32
    wraps points captured just *before* the nominal scan start to ~2^32, which
    reads as several seconds into a 0.1 s scan; reinterpreting those as signed
    restores the small negative offset they actually are.
    """
    topic = src["out"]
    vals = np.asarray(raw)
    if vals.size == 0:
        return vals.astype(np.float64)
    mode = src.get("time_outliers", "keep")
    if mode == "signed" and vals.dtype.kind == "u":
        bits = vals.dtype.itemsize * 8
        wrapped = vals >= (1 << (bits - 1))
        if wrapped.any():
            warn_once(f"{topic}:wrap", (
                f"per-point '{field_name}' has values >= 2^{bits - 1} (unsigned "
                f"underflow); reinterpreting as signed, i.e. as small negative "
                f"offsets from before the scan start"))
            vals = vals.astype(np.int64)
            vals[wrapped] -= 1 << bits
    secs = to_seconds(np.asarray(vals, dtype=np.float64), unit)
    span_ns = float(secs.max() - secs.min()) * 1e9
    msg = span_warning(span_ns, field_name, unit)
    if msg and path:
        msg = f"{msg}  [first seen in {os.path.basename(path)}]"
    warn_once(f"{topic}:time", msg)
    if epoch_kind(float(secs.min()) * 1e9) == "unix_absolute":
        return secs - stamp_ns / 1e9
    return secs


def read_livox_bin(path: str, src: dict[str, Any], stamp_ns: int) -> dict[str, np.ndarray]:
    pts = np.fromfile(path, dtype=LIVOX_BIN_DTYPE)
    return {
        "x": pts["x"], "y": pts["y"], "z": pts["z"],
        "intensity": pts["refl"].astype(np.float32),
        "time": relative_seconds(pts["offset"], src.get("time_unit", "ns"), stamp_ns,
                                 src, "offset", path),
        "tag": pts["tag"], "line": pts["line"],
        # Livox calls the beam index `line`; it is the `ring` of the Velodyne
        # layout (and is already listed in pc2_layout.DEFAULT_RING_FIELDS), so
        # expose it under both names and a Livox dataset can feed either layout.
        "ring": pts["line"],
    }


def read_xyzi_bin(path: str, src: dict[str, Any], stamp_ns: int) -> dict[str, np.ndarray]:
    """Flat float32 records (KITTI-style ``x y z intensity``, or any N columns)."""
    names = src.get("bin_columns", ["x", "y", "z", "intensity"])
    arr = np.fromfile(path, dtype=src.get("bin_dtype", "<f4")).reshape(-1, len(names))
    return {n: arr[:, i] for i, n in enumerate(names) if n != "_"}


def read_pcd(path: str, src: dict[str, Any], stamp_ns: int) -> dict[str, np.ndarray]:
    """Any PCD, ascii or binary (parsed by ``pcd_io``, shared with ``pcd_convert.py``).
    All fields are kept, then aliased to canonical names using the same vendor
    tables ``inspect_bag.py`` probes with."""
    data = parse_pcd(path)
    out: dict[str, np.ndarray] = {}
    for axis in ("x", "y", "z"):
        if axis not in data:
            raise SystemExit(f"[{src['out']}] PCD {path} has no '{axis}' field; "
                             f"present: {sorted(data)}")
        out[axis] = data[axis]

    iname = pick_field(data, src.get("intensity_field"), DEFAULT_INTENSITY_FIELDS)
    if iname:
        out["intensity"] = data[iname]
    tname = pick_field(data, src.get("time_field"), DEFAULT_TIME_FIELDS)
    if tname:
        out["time"] = relative_seconds(data[tname], src.get("time_unit", "s"),
                                       stamp_ns, src, tname, path)
    rname = pick_field(data, src.get("ring_field"), DEFAULT_RING_FIELDS)
    if rname:
        out["ring"] = data[rname]
    return out


READERS = {
    "conpr_livox_bin": read_livox_bin,
    "xyzi_bin": read_xyzi_bin,
    "pcd": read_pcd,
}


def read_points(path: str, src: dict[str, Any], stamp_ns: int) -> dict[str, np.ndarray] | None:
    """Canonical per-point columns for one frame, or None if the frame is
    discarded by ``time_outliers: drop_frame`` -- callers must handle None."""
    fmt = src.get("format")
    if fmt not in READERS:
        raise SystemExit(f"[{src['out']}] unknown format {fmt!r}; known: {sorted(READERS)}")
    canon = READERS[fmt](path, src, stamp_ns)
    unknown = set(canon) - set(CANONICAL_POINT_FIELDS)
    if unknown:   # a reader mapped something not in the shared vocabulary
        for name in unknown:
            canon.pop(name)
    if src.get("drop_zero_points", True):
        keep = ~((canon["x"] == 0) & (canon["y"] == 0) & (canon["z"] == 0))
        if not keep.all():
            canon = {k: v[keep] for k, v in canon.items()}
    # `time_outliers: drop` removes points whose per-point time could not belong
    # to this scan. Corrupt per-point time is worse than a missing point: a de-skew
    # pass propagates the point by its stated offset, so one bad frame throws a
    # quarter of its returns seconds into the future.
    mode = src.get("time_outliers", "keep")
    if mode in ("drop", "drop_frame") and "time" in canon:
        bound = float(src.get("time_max_sec", SCAN_SPAN_MAX_S))
        keep = np.abs(canon["time"]) <= bound
        if not keep.all():
            # warn_once fires once per topic, so a run that hit one bad frame and
            # a run that hit a hundred look identical. Tally them for the summary.
            tally = OUTLIER_TALLY.setdefault(src["out"], {"frames": 0, "points": 0})
            tally["frames"] += 1
            tally["points"] += int((~keep).sum())
            if mode == "drop_frame":
                warn_once(f"{src['out']}:outliers", (
                    f"discarding frames with per-point times outside +/-{bound} s "
                    f"(first: {os.path.basename(path)}, {int((~keep).sum())} of "
                    f"{keep.size} points) -- see time_outliers in README.md"))
                return None
            warn_once(f"{src['out']}:outliers", (
                f"dropping per-point times outside +/-{bound} s "
                f"(first: {int((~keep).sum())} of {keep.size} points in "
                f"{os.path.basename(path)}) -- NOTE this keeps the rest of the "
                f"frame, which may still carry a second, differently-based time "
                f"origin; `drop_frame` is the safer setting"))
            canon = {k: v[keep] for k, v in canon.items()}
    return canon


# --------------------------------------------------------------------------
# Canonical columns -> output message
# --------------------------------------------------------------------------
def clamp_to_type(values: np.ndarray, code: str, context: str, what: str) -> np.ndarray:
    """Saturate a per-point column into its output type's range, reporting how many
    values needed it.

    An out-of-range cast to a narrow integer *wraps*: ``2**32 + 5`` written to a
    uint32 time column reads back as ``5``, i.e. a point 4.3 s into the scan
    becomes one 5 ns in. Saturating is the same direction the existing lower-bound
    clip already takes (``time_outliers: signed`` recovers negative offsets that a
    uint column cannot hold), and unlike wrapping it leaves an obvious artefact
    plus a warning instead of a plausible wrong number.
    """
    info = np.iinfo(np.dtype(code))
    finite = np.isfinite(values) if values.dtype.kind == "f" else np.ones(values.shape, bool)
    outside = ~finite | (values < info.min) | (values > info.max)
    if outside.any():
        warn_once(context, (
            f"{int(outside.sum())} of {outside.size} {what} value(s) do not fit "
            f"{np.dtype(code).name} [{info.min}, {info.max}] -- clamped (NaN/Inf -> 0). "
            f"Check the field's `type:`/`unit:` and, for per-point time, "
            f"`time_unit` / `time_outliers`"))
        values = np.nan_to_num(values, nan=0.0, posinf=float(info.max), neginf=float(info.min))
    return values.clip(info.min, info.max)


def build_pointcloud2(canon: dict[str, np.ndarray], stamp_ns: int, src: dict[str, Any],
                      fields: list[dict[str, Any]], dtype: np.dtype, step: int):
    PointCloud2 = ROS2_STORE.types["sensor_msgs/msg/PointCloud2"]
    PointField = ROS2_STORE.types["sensor_msgs/msg/PointField"]
    Header = ROS2_STORE.types["std_msgs/msg/Header"]

    n = len(canon["x"])
    out = np.zeros(n, dtype=dtype)
    missing = []
    for f in fields:
        col = canon.get(f["from"])
        if col is None:
            missing.append(f["from"])
            continue
        # canonical `time` is seconds; scale to whatever unit this column holds
        # (float columns default to seconds, integer columns to nanoseconds).
        if f["from"] == "time" and f["unit"] != "s":
            col = col / TIME_UNIT_TO_S[f["unit"]]
        col = np.asarray(col)
        code = PC2_DATATYPES[NAME_TO_DATATYPE[f["type"]]][0]
        if code[0] != "f":
            # An integer column wraps on overflow, so bound it: `ouster`'s uint32
            # `t` holds only 4.29 s of nanoseconds, and `velodyne`'s uint16 `ring`
            # only 65535 beams.
            col = clamp_to_type(col, code, f"{src['out']}:clamp:{f['name']}", f"'{f['name']}'")
        elif col.dtype.kind == "f" and not np.isfinite(col).all():
            warn_once(f"{src['out']}:nonfinite:{f['name']}", (
                f"{int((~np.isfinite(col)).sum())} of {col.size} '{f['name']}' value(s) are "
                f"NaN/Inf and are written through as-is -- a consumer that de-skews by "
                f"'time' will propagate them into the point positions"))
        out[f["name"]] = col
    if missing:
        warn_once(f"{src['out']}:layout", (
            f"layout wants {sorted(set(missing))} but format {src.get('format')!r} does not "
            f"provide it for these files -- written as zeros. De-skew needs 'time'; "
            f"vertical downsampling needs 'ring'."
        ))
    # A present-but-constant ring is as useless as a missing one, and just as
    # silent -- mirrors the span-0 case span_warning() already catches for time.
    if "ring" in out.dtype.names and n and len(np.unique(out["ring"])) == 1:
        warn_once(f"{src['out']}:ring", (
            f"'ring' is constant ({int(out['ring'][0])}) -- the source carries no beam "
            f"index, so consumers that downsample vertically by ring will see one layer"
        ))

    return PointCloud2(
        header=Header(stamp=ns_to_time(stamp_ns), frame_id=src.get("frame_id", "")),
        height=1, width=n,
        fields=[PointField(name=f["name"], offset=f["offset"],
                           datatype=NAME_TO_DATATYPE[f["type"]], count=1) for f in fields],
        is_bigendian=False, point_step=step, row_step=step * n,
        data=out.view(np.uint8).reshape(-1), is_dense=True,
    )


# --------------------------------------------------------------------------
# CSV tables (IMU / ground truth)
# --------------------------------------------------------------------------
def load_table(seq_dir: str, src: dict[str, Any], known: tuple[str, ...]
               ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load a CSV into {canonical column name: array} using ``columns:``.

    ``columns`` names each column positionally ("_" skips one), which is what
    makes an 11-column ConPR IMU and a 7-column EuRoC IMU the same code path.
    """
    path = resolve_path(seq_dir, src["path"])
    if not os.path.isfile(path):
        raise SystemExit(f"[{src['out']}] file not found: {path}")
    data = np.loadtxt(path, delimiter=src.get("delimiter", ","),
                      skiprows=int(src.get("skip_rows", 0)), ndmin=2)
    names = src.get("columns")
    if not names:
        raise SystemExit(f"[{src['out']}] `columns:` is required; name each column of "
                         f"{os.path.basename(path)} in order, from {list(known)} ('_' to skip)")
    if len(names) > data.shape[1]:
        raise SystemExit(f"[{src['out']}] `columns:` lists {len(names)} names but "
                         f"{os.path.basename(path)} has {data.shape[1]} columns")
    unknown = [n for n in names if n not in known and n != "_"]
    if unknown:
        raise SystemExit(f"[{src['out']}] unknown column name(s) {unknown}; known: {list(known)}")
    cols = {n: data[:, i] for i, n in enumerate(names) if n != "_"}
    if "t" not in cols:
        raise SystemExit(f"[{src['out']}] `columns:` must include 't' (the timestamp)")
    # np.loadtxt gave float64; re-read the timestamp column as int64 when it is
    # integral, so ns Unix stamps are not rounded to the nearest ~256 ns.
    t_col = read_int_column(path, src.get("delimiter", ","),
                            int(src.get("skip_rows", 0)), names.index("t"))
    stamps = stamps_to_ns(cols["t"] if t_col is None else t_col,
                          src.get("stamp_unit", "ns"), shift_ns_of(src), src["out"])
    check_epoch(stamps, src["out"], "row")
    # A NaN elsewhere in the row is a data defect, but a recoverable one -- the bag
    # is still readable and inspectable -- so report it rather than aborting a
    # 12-sequence batch over one bad accelerometer sample.
    for name, col in cols.items():
        if name != "t" and not np.isfinite(col).all():
            warn_once(f"{src['out']}:nonfinite:{name}", (
                f"column {name!r} has {int((~np.isfinite(col)).sum())} NaN/Inf value(s) of "
                f"{col.size}; they are written through as-is"))
    order = np.argsort(stamps, kind="stable")
    return {k: v[order] for k, v in cols.items()}, stamps[order]


def build_imu(cols: dict[str, np.ndarray], i: int, stamp_ns: int, src: dict[str, Any]):
    Imu = ROS2_STORE.types["sensor_msgs/msg/Imu"]
    Header = ROS2_STORE.types["std_msgs/msg/Header"]
    Vector3 = ROS2_STORE.types["geometry_msgs/msg/Vector3"]
    Quaternion = ROS2_STORE.types["geometry_msgs/msg/Quaternion"]

    def get(name: str, default: float = 0.0) -> float:
        col = cols.get(name)
        return default if col is None else float(col[i])

    zeros9 = np.zeros(9, dtype=np.float64)
    return Imu(
        header=Header(stamp=ns_to_time(stamp_ns), frame_id=src.get("frame_id", "")),
        orientation=Quaternion(x=get("qx"), y=get("qy"), z=get("qz"), w=get("qw", 1.0)),
        orientation_covariance=zeros9.copy(),
        angular_velocity=Vector3(x=get("wx"), y=get("wy"), z=get("wz")),
        angular_velocity_covariance=zeros9.copy(),
        linear_acceleration=Vector3(x=get("ax"), y=get("ay"), z=get("az")),
        linear_acceleration_covariance=zeros9.copy(),
    )


def build_odometry(cols: dict[str, np.ndarray], i: int, stamp_ns: int, src: dict[str, Any],
                   quat: tuple[float, float, float, float]):
    Odometry = ROS2_STORE.types["nav_msgs/msg/Odometry"]
    Header = ROS2_STORE.types["std_msgs/msg/Header"]
    Point = ROS2_STORE.types["geometry_msgs/msg/Point"]
    Quaternion = ROS2_STORE.types["geometry_msgs/msg/Quaternion"]
    Pose = ROS2_STORE.types["geometry_msgs/msg/Pose"]
    PoseWithCovariance = ROS2_STORE.types["geometry_msgs/msg/PoseWithCovariance"]
    Twist = ROS2_STORE.types["geometry_msgs/msg/Twist"]
    TwistWithCovariance = ROS2_STORE.types["geometry_msgs/msg/TwistWithCovariance"]
    Vector3 = ROS2_STORE.types["geometry_msgs/msg/Vector3"]

    qx, qy, qz, qw = quat
    return Odometry(
        header=Header(stamp=ns_to_time(stamp_ns), frame_id=src.get("frame_id", "map")),
        child_frame_id=src.get("child_frame_id", ""),
        pose=PoseWithCovariance(
            pose=Pose(position=Point(x=float(cols["x"][i]), y=float(cols["y"][i]),
                                     z=float(cols["z"][i])),
                      orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)),
            covariance=np.zeros(36, dtype=np.float64)),
        twist=TwistWithCovariance(
            twist=Twist(linear=Vector3(x=0.0, y=0.0, z=0.0),
                        angular=Vector3(x=0.0, y=0.0, z=0.0)),
            covariance=np.zeros(36, dtype=np.float64)),
    )


def nearest_index(stamps_ns: np.ndarray, stamp_ns: int) -> int:
    return int(np.abs(stamps_ns - stamp_ns).argmin())


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def prepare(seq_dir: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the sources list and load everything except point files."""
    if "sources" not in cfg:
        hint = ("  this config uses the old `dataset:`/`bag:`/`frames:` schema; the "
                "converter now takes a single `sources:` list -- see README.md"
                ) if ("dataset" in cfg or "bag" in cfg) else ""
        raise SystemExit(f"config has no `sources:` list\n{hint}")

    prepared = []
    for raw in cfg["sources"]:
        src = dict(raw)
        if not src.get("enabled", True):
            continue
        if "out" not in src:
            raise SystemExit(f"every source needs an `out:` topic; got {sorted(src)}")
        src["out"] = normalize_topic(src["out"])
        typ = src.get("type")
        if typ not in SOURCE_TYPES:
            raise SystemExit(f"[{src['out']}] unknown type {typ!r}; known: {list(SOURCE_TYPES)}")

        if typ in CLOUD_TYPES:
            src["paths"], src["stamps"] = list_frames(seq_dir, src)
            spec = src.get("fields") or src.get("layout", "xyzi")
            fields = normalize_layout(spec, src["out"])
            src["dtype"], src["fields_r"], src["step"] = build_layout(
                fields, src.get("point_step"))
            src["msgtype"] = "sensor_msgs/msg/PointCloud2"
        elif typ == "imu":
            src["cols"], src["stamps"] = load_table(seq_dir, src, IMU_COLUMNS)
            src["msgtype"] = "sensor_msgs/msg/Imu"
        else:
            src["cols"], src["stamps"] = load_table(seq_dir, src, ODOM_COLUMNS)
            src["msgtype"] = "nav_msgs/msg/Odometry"
            for axis in ("x", "y", "z"):
                if axis not in src["cols"]:
                    raise SystemExit(f"[{src['out']}] odometry needs columns x, y, z")
        prepared.append(src)

    if not prepared:
        raise SystemExit("no enabled sources in config")
    dupes = {s["out"] for s in prepared if [x["out"] for x in prepared].count(s["out"]) > 1}
    if dupes:
        raise SystemExit(f"duplicate output topics: {sorted(dupes)}")

    # Resolve cross-source references now that every topic is known.
    by_topic = {s["out"]: s for s in prepared}
    for src in prepared:
        if src["type"] != "odometry":
            continue
        at = src.get("publish_at", "rows")
        if at != "rows":
            ref = by_topic.get(normalize_topic(at))
            if ref is None:
                raise SystemExit(f"[{src['out']}] publish_at: {at!r} is not one of "
                                 f"{sorted(by_topic)} (or 'rows')")
            src["publish_stamps"] = ref["stamps"]
        orient = src.get("orientation", "columns")
        if orient == "nearest_imu":
            imus = [s for s in prepared if s["type"] == "imu"]
            if len(imus) != 1:
                raise SystemExit(f"[{src['out']}] orientation: nearest_imu needs exactly one "
                                 f"imu source, found {len(imus)}")
            src["imu_ref"] = imus[0]
        elif orient != "columns":
            raise SystemExit(f"[{src['out']}] unknown orientation {orient!r}; "
                             f"use columns|nearest_imu")
        if orient == "columns" and "qw" not in src["cols"]:
            warn_once(f"{src['out']}:orient", (
                "no quaternion columns given, orientation will be identity; use "
                "`orientation: nearest_imu` to borrow the IMU attitude"))
    return prepared


def build_events(prepared: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """(write_stamp_ns, source index, item index) for everything, sorted.

    ``min_dt_sec`` decimates on sensor time; ``write_stamp_shift_sec`` moves only
    the bag write stamp (not the header), the knob that makes a streaming
    consumer see a full scan's worth of IMU before the scan itself.
    """
    events: list[tuple[int, int, int]] = []
    for si, src in enumerate(prepared):
        if src["type"] == "odometry" and "publish_stamps" in src:
            stamps = src["publish_stamps"]
        else:
            stamps = src["stamps"]
        min_dt = int(round(float(src.get("min_dt_sec", 0.0)) * 1e9))
        shift = int(round(float(src.get("write_stamp_shift_sec", 0.0)) * 1e9))
        last = None
        for i, t in enumerate(stamps):
            t = int(t)
            if min_dt > 0 and last is not None and t - last < min_dt:
                continue
            last = t
            events.append((t + shift, si, i))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return events


def convert(seq_dir: str, out_bag: str, cfg: dict[str, Any],
            start_sec: float | None = None, limit_sec: float | None = None,
            force: bool = False) -> None:
    # Warn-once and outlier bookkeeping are module state, so a second conversion in
    # the same process would otherwise inherit the first one's "already reported".
    reset_warnings()
    OUTLIER_TALLY.clear()

    prepared = prepare(seq_dir, cfg)
    events = build_events(prepared)
    if not events:
        raise SystemExit("nothing to write: every source is empty")

    t0 = events[0][0]
    counts = {src["out"]: 0 for src in prepared}
    # atomic_bag writes to a sibling staging directory and only swaps it into place
    # once this block completes, so a read error, a full disk or a Ctrl-C leaves any
    # previous bag untouched instead of half-replaced.
    with atomic_bag(out_bag, cfg, force) as writer:
        conns = {}
        for si, src in enumerate(prepared):
            conns[si] = writer.add_connection(src["out"], src["msgtype"], typestore=ROS2_STORE)

        tf_static = cfg.get("tf_static")
        if tf_static:
            counts["/tf_static"] = write_tf_static(writer, tf_static, t0)

        for write_ns, si, i in events:
            rel = (write_ns - t0) / 1e9
            if start_sec is not None and rel < start_sec:
                continue
            if limit_sec is not None and rel > limit_sec:
                break
            src = prepared[si]
            shift = int(round(float(src.get("write_stamp_shift_sec", 0.0)) * 1e9))
            header_ns = write_ns - shift   # the header keeps sensor time

            if src["type"] in CLOUD_TYPES:
                canon = read_points(src["paths"][i], src, header_ns)
                if canon is None:       # time_outliers: drop_frame discarded it
                    continue
                msg = build_pointcloud2(canon, header_ns, src, src["fields_r"],
                                        src["dtype"], src["step"])
            elif src["type"] == "imu":
                msg = build_imu(src["cols"], i, header_ns, src)
            else:
                cols = src["cols"]
                row = i if "publish_stamps" not in src else nearest_index(src["stamps"], header_ns)
                if "imu_ref" in src:
                    ref = src["imu_ref"]
                    j = nearest_index(ref["stamps"], header_ns)
                    rc = ref["cols"]
                    quat = (float(rc.get("qx", [0])[j]) if "qx" in rc else 0.0,
                            float(rc.get("qy", [0])[j]) if "qy" in rc else 0.0,
                            float(rc.get("qz", [0])[j]) if "qz" in rc else 0.0,
                            float(rc.get("qw", [1])[j]) if "qw" in rc else 1.0)
                elif "qw" in cols:
                    quat = (float(cols["qx"][row]), float(cols["qy"][row]),
                            float(cols["qz"][row]), float(cols["qw"][row]))
                else:
                    quat = (0.0, 0.0, 0.0, 1.0)
                msg = build_odometry(cols, row, header_ns, src, quat)

            writer.write(conns[si], write_ns, ROS2_STORE.serialize_cdr(msg, src["msgtype"]))
            counts[src["out"]] += 1

    print(f"done: {', '.join(f'{n} {t}' for t, n in counts.items())} -> {out_bag} "
          f"[{output_summary(cfg)}]")
    for topic, tally in OUTLIER_TALLY.items():
        mode = next((s.get("time_outliers") for s in prepared if s["out"] == topic), "")
        written = counts.get(topic, 0)
        if mode == "drop_frame":
            print(f"  [{topic}] discarded {tally['frames']} corrupt-time frames "
                  f"({tally['points']} points); wrote {written}")
        else:
            print(f"  [{topic}] dropped {tally['points']} corrupt-time points across "
                  f"{tally['frames']} of {written} frames")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("seq_dir", help="sequence directory")
    parser.add_argument("-o", "--out", "--output", dest="output", required=True,
                        help="output ROS 2 bag directory")
    parser.add_argument("-c", "--config", required=True, help="conversion YAML config")
    parser.add_argument("--start-sec", type=float, default=None,
                        help="skip messages before N seconds from the first message")
    parser.add_argument("--limit-sec", type=float, default=None,
                        help="stop at N seconds from the first message")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing output bag (only once the new one is "
                             "complete -- a failed conversion leaves the old bag intact)")
    # Opt-in here, unlike batch_to_bag.py where it is on by default: the batch
    # driver samples the whole process tree, so a default-on flag here would have
    # every subprocess writing a redundant log of its own slice of the same run.
    parser.add_argument("--resource-log", metavar="CSV",
                        help="log CPU/memory/disk usage to this CSV and draw a chart "
                             "beside it (see src/resource_log.py)")
    parser.add_argument("--resource-interval", type=float, default=1.0, metavar="SEC",
                        help="resource sampling period (default: 1.0)")
    add_distro_arg(parser)
    args = parser.parse_args(argv)
    check_window(args.start_sec, args.limit_sec)
    if os.path.exists(args.output) and not args.force:
        raise SystemExit(f"output already exists: {args.output}")
    cfg = apply_distro(load_yaml(args.config), args.ros_distro)
    with resource_log.record(args.resource_log, args.resource_interval,
                             disk_path=os.path.dirname(os.path.abspath(args.output))):
        convert(args.seq_dir, args.output, cfg, args.start_sec, args.limit_sec, args.force)


if __name__ == "__main__":
    main()
