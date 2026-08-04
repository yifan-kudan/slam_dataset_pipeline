#!/usr/bin/env python3
"""
Config-driven converter from dataset folders to ROS 2 bags.

Companion to ``rosbag1_to_ros2.py``: that one re-containers an existing ROS 1
recording, this one builds a bag from loose per-frame LiDAR files plus CSV
IMU/ground-truth tables. They share the output half entirely -- typestore,
``output:`` keys, ``tf_static:`` schema, per-topic knobs (``frame_id``,
``min_dt_sec``, ``write_stamp_shift_sec``), the ``time_field``/``time_unit``/
``ring_field`` vocabulary, and the named cloud layouts in ``pc2_layout`` -- so a
bag built either way is interchangeable downstream.

Config shape (see to_bag_config.md)::

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

Runs on pure-python ``rosbags`` -- no ROS environment needs to be sourced, not
even for Livox CustomMsg output.
"""
import argparse
import glob
import os
from typing import Any

import numpy as np

from pc2_layout import (
    CANONICAL_POINT_FIELDS, DEFAULT_INTENSITY_FIELDS, DEFAULT_RING_FIELDS,
    DEFAULT_TIME_FIELDS, LAYOUT_TIME_SCALE, NAME_TO_DATATYPE, TIME_UNIT_TO_NS,
    build_layout, epoch_kind, normalize_layout, span_warning, to_seconds, warn_once,
)
from ros2_bag_io import (
    ROS2_STORE, load_yaml, ns_to_time, normalize_topic, open_writer,
    register_livox_types, write_tf_static,
)

SOURCE_TYPES = ("pointcloud2", "livox_custom", "imu", "odometry")
CLOUD_TYPES = ("pointcloud2", "livox_custom")

# Canonical CSV column names understood by `columns:`. "_" skips a column.
IMU_COLUMNS = ("t", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az")
ODOM_COLUMNS = ("t", "x", "y", "z", "qx", "qy", "qz", "qw")

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


def stamps_to_ns(values: np.ndarray, unit: str, shift_ns: int) -> np.ndarray:
    """Whole-message timestamps -> int64 nanoseconds.

    Scaling in float64 then rounding: a Unix-nanosecond value (~1.7e18) exceeds
    float64's 2**53 exact-integer range, so for the common `unit: ns` case the
    values are kept integral instead of being routed through the multiply.
    """
    if unit == "ns":
        ns = np.asarray(values, dtype=np.float64).round().astype(np.int64)
    else:
        ns = np.rint(np.asarray(values, dtype=np.float64) * stamp_scale(unit)).astype(np.int64)
    return ns + np.int64(shift_ns)


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
        raw = []
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                raw.append(float(stem))
            except ValueError:
                raise SystemExit(
                    f"[{src['out']}] filename {os.path.basename(p)!r} is not a number, so it "
                    f"carries no timestamp; point `stamp_file:` at the dataset's timestamp "
                    f"list and set `stamp_from: file`"
                ) from None
        stamps = stamps_to_ns(np.array(raw, dtype=np.float64), unit, shift)
    elif stamp_from == "file":
        if "stamp_file" not in src:
            raise SystemExit(f"[{src['out']}] stamp_from: file requires `stamp_file:`")
        table = np.loadtxt(resolve_path(seq_dir, src["stamp_file"]), ndmin=2,
                           delimiter=src.get("stamp_delimiter"))
        col = int(src.get("stamp_column", 0))
        if len(table) != len(paths):
            raise SystemExit(
                f"[{src['out']}] stamp_file has {len(table)} rows but {len(paths)} files "
                f"match {src.get('pattern', '*')!r} -- they must correspond one to one"
            )
        stamps = stamps_to_ns(table[:, col], unit, shift)
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


def relative_seconds(raw: np.ndarray, unit: str, stamp_ns: int, topic: str,
                     field_name: str) -> np.ndarray:
    """Per-point time column -> seconds relative to the frame stamp.

    Handles both conventions seen in the wild: scan-relative offsets (Livox,
    Velodyne) pass through scaled, absolute Unix time (Hesai) is differenced
    against the frame stamp -- the same choice ``repack_hesai_to_velodyne`` makes
    for ROS 1 input. The observed span is checked so a wrong ``time_unit``, which
    would otherwise scale silently by a power of 1000, gets a warning.
    """
    vals = np.asarray(raw, dtype=np.float64)
    if vals.size == 0:
        return vals.astype(np.float64)
    secs = to_seconds(vals, unit)
    span_ns = float(secs.max() - secs.min()) * 1e9
    warn_once(f"{topic}:time", span_warning(span_ns, field_name, unit))
    if epoch_kind(float(secs.min()) * 1e9) == "unix_absolute":
        return secs - stamp_ns / 1e9
    return secs


def read_livox_bin(path: str, src: dict[str, Any], stamp_ns: int) -> dict[str, np.ndarray]:
    pts = np.fromfile(path, dtype=LIVOX_BIN_DTYPE)
    return {
        "x": pts["x"], "y": pts["y"], "z": pts["z"],
        "intensity": pts["refl"].astype(np.float32),
        "time": relative_seconds(pts["offset"], src.get("time_unit", "ns"), stamp_ns,
                                 src["out"], "offset"),
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
    """Any PCD, ascii or binary. All fields are kept, then aliased to canonical
    names using the same vendor tables ``inspect_bag.py`` probes with."""
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
                                       stamp_ns, src["out"], tname)
    rname = pick_field(data, src.get("ring_field"), DEFAULT_RING_FIELDS)
    if rname:
        out["ring"] = data[rname]
    return out


def parse_pcd(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"PCD header missing DATA line: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text.startswith("DATA"):
                data_kind = text.split()[1].lower()
                break
        header = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            header[parts[0].upper()] = parts[1:]
        fields = header["FIELDS"]
        sizes = [int(x) for x in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
        width = int(header.get("WIDTH", ["0"])[0])
        height = int(header.get("HEIGHT", ["1"])[0])
        points = int(header.get("POINTS", [str(width * height)])[0])
        if any(c != 1 for c in counts):
            raise SystemExit(f"PCD fields with COUNT != 1 are not supported yet: {path}")
        dtype_fields = []
        for name, size, typ in zip(fields, sizes, types):
            code = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2",
                    ("U", 4): "<u4", ("U", 8): "<u8", ("I", 1): "i1", ("I", 2): "<i2",
                    ("I", 4): "<i4", ("I", 8): "<i8"}.get((typ, size))
            if code is None:
                raise SystemExit(f"unsupported PCD field type {typ}{size} in {path}")
            dtype_fields.append((name, code))
        dtype = np.dtype(dtype_fields)
        if data_kind == "ascii":
            arr = np.loadtxt(f, dtype=np.float64, ndmin=2)
            return {name: arr[:, i] for i, name in enumerate(fields)}
        if data_kind == "binary":
            arr = np.frombuffer(f.read(dtype.itemsize * points), dtype=dtype, count=points)
            return {name: arr[name] for name in fields}
        raise SystemExit(f"unsupported PCD DATA mode {data_kind}: {path}; "
                         f"convert binary_compressed with pcl_convert_pcd_ascii_binary")


READERS = {
    "conpr_livox_bin": read_livox_bin,
    "xyzi_bin": read_xyzi_bin,
    "pcd": read_pcd,
}


def read_points(path: str, src: dict[str, Any], stamp_ns: int) -> dict[str, np.ndarray]:
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
    return canon


# --------------------------------------------------------------------------
# Canonical columns -> output message
# --------------------------------------------------------------------------
def build_pointcloud2(canon: dict[str, np.ndarray], stamp_ns: int, src: dict[str, Any],
                      fields: list[dict[str, Any]], dtype: np.dtype, step: int):
    PointCloud2 = ROS2_STORE.types["sensor_msgs/msg/PointCloud2"]
    PointField = ROS2_STORE.types["sensor_msgs/msg/PointField"]
    Header = ROS2_STORE.types["std_msgs/msg/Header"]

    n = len(canon["x"])
    out = np.zeros(n, dtype=dtype)
    scale = LAYOUT_TIME_SCALE.get(src.get("layout"), 1.0)
    missing = []
    for f in fields:
        col = canon.get(f["from"])
        if col is None:
            missing.append(f["from"])
            continue
        out[f["name"]] = col * scale if f["from"] == "time" and scale != 1.0 else col
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


def build_livox_custom(canon: dict[str, np.ndarray], stamp_ns: int, src: dict[str, Any]):
    """Livox CustomMsg, the format FAST-LIO's livox_handler reads directly.

    ``offset_time`` is nanoseconds from ``timebase`` per the livox_ros_driver2
    definition, which is also how FAST-LIO reads it (it divides by 1e6 to get
    milliseconds). ``offset_time_unit`` exists only to reproduce bags that were
    written with a different convention.
    """
    register_livox_types()
    CustomMsg = ROS2_STORE.types["livox_ros_driver2/msg/CustomMsg"]
    CustomPoint = ROS2_STORE.types["livox_ros_driver2/msg/CustomPoint"]
    Header = ROS2_STORE.types["std_msgs/msg/Header"]

    n = len(canon["x"])
    unit = src.get("offset_time_unit", "ns")
    if unit not in TIME_UNIT_TO_NS:
        raise SystemExit(f"[{src['out']}] unknown offset_time_unit {unit!r}; "
                         f"expected one of {sorted(TIME_UNIT_TO_NS)}")
    # canonical `time` is seconds -> the requested output unit.
    offs = np.rint(canon.get("time", np.zeros(n)) * (1e9 / TIME_UNIT_TO_NS[unit]))
    # Build from plain python lists -- much faster than indexing arrays per point.
    ox = offs.clip(0).astype(np.uint32).tolist()
    xs, ys, zs = (canon[a].astype(np.float32).tolist() for a in ("x", "y", "z"))
    refl = canon.get("intensity", np.zeros(n)).clip(0, 255).astype(np.uint8).tolist()
    tags = canon.get("tag", np.zeros(n, np.uint8)).astype(np.uint8).tolist()
    lines = canon.get("line", np.zeros(n, np.uint8)).astype(np.uint8).tolist()

    return CustomMsg(
        header=Header(stamp=ns_to_time(stamp_ns), frame_id=src.get("frame_id", "")),
        timebase=int(stamp_ns), point_num=n,
        lidar_id=int(src.get("lidar_id", 0)),
        rsvd=np.array(src.get("rsvd", [0, 0, 0]), dtype=np.uint8),
        points=[CustomPoint(offset_time=o, x=x, y=y, z=z, reflectivity=r, tag=g, line=l)
                for o, x, y, z, r, g, l in zip(ox, xs, ys, zs, refl, tags, lines)],
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
    stamps = stamps_to_ns(cols["t"], src.get("stamp_unit", "ns"), shift_ns_of(src))
    check_epoch(stamps, src["out"], "row")
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
                "converter now takes a single `sources:` list -- see to_bag_config.md"
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
            if typ == "pointcloud2":
                spec = src.get("fields") or src.get("layout", "xyzi")
                src["layout"] = spec if isinstance(spec, str) else src.get("layout")
                fields = normalize_layout(spec, src["out"])
                src["dtype"], src["fields_r"], src["step"] = build_layout(
                    fields, src.get("point_step"))
                src["msgtype"] = "sensor_msgs/msg/PointCloud2"
            else:
                src["msgtype"] = "livox_ros_driver2/msg/CustomMsg"
                register_livox_types()
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
            start_sec: float | None = None, limit_sec: float | None = None) -> None:
    prepared = prepare(seq_dir, cfg)
    events = build_events(prepared)
    if not events:
        raise SystemExit("nothing to write: every source is empty")

    t0 = events[0][0]
    writer = open_writer(out_bag, cfg)
    conns = {}
    for si, src in enumerate(prepared):
        conns[si] = writer.add_connection(src["out"], src["msgtype"], typestore=ROS2_STORE)

    counts = {src["out"]: 0 for src in prepared}
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
            if src["type"] == "pointcloud2":
                msg = build_pointcloud2(canon, header_ns, src, src["fields_r"],
                                        src["dtype"], src["step"])
            else:
                msg = build_livox_custom(canon, header_ns, src)
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

    writer.close()
    print(f"done: {', '.join(f'{n} {t}' for t, n in counts.items())} -> {out_bag}")


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
    args = parser.parse_args(argv)
    if args.start_sec is not None and args.limit_sec is not None and args.start_sec >= args.limit_sec:
        raise SystemExit(f"--start-sec ({args.start_sec}) must be < --limit-sec ({args.limit_sec})")
    if os.path.exists(args.output):
        raise SystemExit(f"output already exists: {args.output}")
    convert(args.seq_dir, args.output, load_yaml(args.config), args.start_sec, args.limit_sec)


if __name__ == "__main__":
    main()
