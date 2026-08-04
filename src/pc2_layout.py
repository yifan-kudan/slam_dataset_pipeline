"""Shared PointCloud2 layout helpers for the bag tools.

Splits the PointCloud2 handling into the part that is **fixed by the ROS message
spec** and the part that **varies per dataset**, so the varying part can be
configured instead of edited:

fixed
  * ``PC2_DATATYPES`` -- the ``sensor_msgs/PointField.datatype`` enum. Values
    1..8 (INT8..FLOAT64), identical in ROS 1 and ROS 2, so any spec-conforming
    cloud is covered.
  * byte order -- taken from each message's own ``is_bigendian`` flag rather than
    hardcoded little-endian.

varies per dataset (configurable; defaults below are just the common cases)
  * field *names*
  * field *units*
  * absolute vs scan-relative per-point time (Hesai ships absolute Unix seconds).
    ``epoch_kind`` classifies a stamp; the converter's ``repack`` handles it.

Used by ``inspect_bag.py``, ``rosbag1_to_ros2.py`` and ``to_bag.py``.
"""
from typing import Any

import numpy as np

# sensor_msgs/PointField datatype enum -> (numpy base code, human name).
# Fixed by the message spec (see /opt/ros/<distro>/share/sensor_msgs/msg/
# PointField.msg): INT8=1 .. FLOAT64=8. No byte-order prefix here -- it comes
# from the message's is_bigendian flag, see numpy_code().
PC2_DATATYPES: dict[int, tuple[str, str]] = {
    1: ("i1", "int8"), 2: ("u1", "uint8"), 3: ("i2", "int16"), 4: ("u2", "uint16"),
    5: ("i4", "int32"), 6: ("u4", "uint32"), 7: ("f4", "float32"), 8: ("f8", "float64"),
}

# Per-point time / ring field names, by vendor.
DEFAULT_TIME_FIELDS: tuple[str, ...] = (
    "time",         # Velodyne / FAST-LIO velodyne_handler
    "t",            # Ouster
    "timestamp",    # Hesai (PandarXT), often absolute Unix seconds
    "time_offset",
    "offset_time",  # Livox
)
DEFAULT_RING_FIELDS: tuple[str, ...] = ("ring", "channel", "line", "laser_id")
DEFAULT_INTENSITY_FIELDS: tuple[str, ...] = (
    "intensity",
    "i",
    "reflectivity",  # Livox / Ouster
    "reflectance",
    "refl",
)

# Canonical per-point quantity names used to decouple *readers* from *writers*:
# every loader in to_bag.py emits these names, and every output layout below is
# expressed in terms of them, so an input format and an output layout can be
# paired without either knowing about the other.
#
#   x, y, z    metres
#   intensity  whatever the sensor reports
#   time       SECONDS since the scan start (relative), matching the Velodyne
#              convention that repack_hesai_to_velodyne() also targets
#   ring       laser/beam index
#   tag, line  Livox-specific per-point flags
CANONICAL_POINT_FIELDS: tuple[str, ...] = (
    "x", "y", "z", "intensity", "time", "ring", "tag", "line",
)

# Per-point time units seen in the wild -> multiplier to nanoseconds / to seconds.
# Ordered coarse->fine; infer_time_unit() relies on that order. Both tables are
# spelled out rather than deriving one from the other: absolute Unix values are
# ~1.7e9 s, and routing them through nanoseconds (*1e9 then /1e9) costs ~130 ns of
# float64 precision.
TIME_UNIT_TO_NS: dict[str, float] = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
TIME_UNIT_TO_S: dict[str, float] = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}

# One LiDAR revolution at 5-20 Hz is 0.05-0.2 s. Allow an order of magnitude
# either side before calling an observed span implausible for a scan.
SCAN_SPAN_MIN_S, SCAN_SPAN_MAX_S = 0.005, 2.0

# Plausible-Unix-time window (ns): 2001-09-09 .. 2033-05-18.
UNIX_LO_NS, UNIX_HI_NS = 1_000_000_000_000_000_000, 2_000_000_000_000_000_000


def type_name(datatype: int) -> str:
    """Readable name for a PointField datatype enum value."""
    entry = PC2_DATATYPES.get(datatype)
    return entry[1] if entry else f"datatype{datatype}"


def numpy_code(datatype: int, is_bigendian: bool = False) -> str:
    """PointField datatype -> numpy dtype string, honouring the cloud's byte order.

    Single-byte types get no prefix (byte order is meaningless for them).
    """
    base, _ = PC2_DATATYPES[datatype]
    if base.endswith("1"):
        return base
    return (">" if is_bigendian else "<") + base


def find_field(msg: Any, candidates: tuple[str, ...] | list[str]) -> Any | None:
    """First PointField in ``msg`` whose name is in ``candidates``."""
    by_name = {f.name: f for f in msg.fields}
    for name in candidates:
        if name in by_name:
            return by_name[name]
    return None


def field_view(buf: Any, msg: Any, field: Any, strict: bool = False) -> np.ndarray:
    """PointCloud2 stores points interleaved, so field *i* of point *k* lives at
    ``k * point_step + field.offset``; a numpy view with ``strides=(point_step,)``
    reads the whole column without unpacking in Python.

    ``buf`` is supplied by the caller: pass ``bytes(msg.data)`` for a read-only
    view, ``bytearray(msg.data)`` when the column will be written in place.
    ``strict`` rejects ``count != 1`` fields (the view would only cover their
    first element) -- converters should set it, probes should not.
    """
    if int(field.count) != 1 and strict:
        raise SystemExit(
            f"field '{field.name}' has count={field.count}; only scalar (count=1) "
            f"fields can be rewritten in place"
        )
    n = int(msg.width) * int(msg.height)
    code = numpy_code(field.datatype, bool(getattr(msg, "is_bigendian", False)))
    return np.ndarray(shape=(n,), dtype=code, buffer=buf,
                      offset=int(field.offset), strides=(int(msg.point_step),))


def to_ns(value: float, unit: str) -> float:
    """Convert a per-point time value in ``unit`` to nanoseconds."""
    return value * _scale(TIME_UNIT_TO_NS, unit)


def to_seconds(value: float, unit: str) -> float:
    """Convert a per-point time value in ``unit`` to seconds.

    Prefer this over ``to_ns(...) / 1e9`` for absolute Unix values -- for
    ``unit="s"`` the factor is exactly 1.0, so the value passes through unchanged.
    """
    return value * _scale(TIME_UNIT_TO_S, unit)


def _scale(table: dict[str, float], unit: str) -> float:
    try:
        return table[unit]
    except KeyError:
        raise SystemExit(
            f"unknown time_unit {unit!r}; expected one of {sorted(table)}"
        ) from None


def infer_time_unit(span: float) -> str | None:
    for unit, scale in TIME_UNIT_TO_NS.items():
        span_s = span * scale / 1e9
        if SCAN_SPAN_MIN_S <= span_s <= SCAN_SPAN_MAX_S:
            return unit
    return None


def span_warning(span_ns: float, field_name: str, unit: str) -> str | None:
    """Warn when a span, read with the configured ``unit``, is not scan-shaped.

    Catches the silent failure mode of a wrong ``time_unit``: reading nanoseconds
    as seconds is off by 1e9 and would corrupt every stamp with no error.
    """
    if span_ns == 0:
        return (f"per-point '{field_name}' is constant (span 0) -- the field carries "
                f"no intra-scan timing, so de-skew will do nothing")
    span_s = span_ns / 1e9
    if not (SCAN_SPAN_MIN_S <= span_s <= SCAN_SPAN_MAX_S):
        better = infer_time_unit(span_ns / TIME_UNIT_TO_NS[unit])
        hint = f"; try time_unit: {better}" if better and better != unit else ""
        return (f"per-point '{field_name}' spans {span_s:.6g} s when read as "
                f"time_unit: {unit} -- not a plausible scan duration "
                f"({SCAN_SPAN_MIN_S}-{SCAN_SPAN_MAX_S} s){hint}")
    return None


def epoch_kind(ns: int | float) -> str:
    """Classify a nanosecond stamp as absolute Unix time or sensor-relative."""
    if UNIX_LO_NS <= ns <= UNIX_HI_NS:
        return "unix_absolute"
    if abs(ns) < UNIX_LO_NS:
        return "sensor_relative"
    return "unknown"


# --------------------------------------------------------------------------
# Warn-once bookkeeping (shared so a per-topic warning fires once per run
# regardless of which converter emitted it).
# --------------------------------------------------------------------------
_warned_topics: set[str] = set()


def warn_once(topic: str, message: str | None) -> None:
    if message and topic not in _warned_topics:
        _warned_topics.add(topic)
        print(f"WARNING [{topic}]: {message}")


def reset_warnings() -> None:
    """Forget which warnings have fired, so a second in-process conversion reports
    its own. Each converter's ``convert()`` calls this on entry; a CLI run (one
    conversion per process) is unaffected."""
    _warned_topics.clear()


# --------------------------------------------------------------------------
# Output cloud layouts
#
# The reader side of a converter is dataset-specific; the *writer* side is
# dictated by what the consuming SLAM node expects, and there are only a handful
# of those. Naming them here gives both converters one vocabulary: `velodyne` is
# byte-for-byte the layout repack_hesai_to_velodyne() produces, so a bag built
# from raw files and a bag repacked from a ROS 1 recording are interchangeable.
#
# Each entry is a list of (output field name, canonical source name, type name).
# Offsets and point_step are computed by build_layout() unless overridden.
# --------------------------------------------------------------------------
NAME_TO_DATATYPE: dict[str, int] = {name: enum for enum, (_, name) in PC2_DATATYPES.items()}

LAYOUT_PRESETS: dict[str, list[tuple[str, str, str]]] = {
    # Bare geometry + intensity. Understood by essentially every consumer, but
    # carries no timing, so motion de-skew is disabled.
    "xyzi": [
        ("x", "x", "float32"), ("y", "y", "float32"), ("z", "z", "float32"),
        ("intensity", "intensity", "float32"),
    ],
    # FAST-LIO velodyne_handler / kd_localizer: float32 `time` seconds relative to
    # the header stamp, uint16 `ring`. Matches repack_hesai_to_velodyne() exactly.
    "velodyne": [
        ("x", "x", "float32"), ("y", "y", "float32"), ("z", "z", "float32"),
        ("intensity", "intensity", "float32"),
        ("time", "time", "float32"), ("ring", "ring", "uint16"),
    ],
    # Ouster driver layout: `t` is uint32 NANOSECONDS from scan start.
    "ouster": [
        ("x", "x", "float32"), ("y", "y", "float32"), ("z", "z", "float32"),
        ("intensity", "intensity", "float32"),
        ("t", "time", "uint32"), ("ring", "ring", "uint16"),
    ],
    # Livox PointCloud2 (not CustomMsg): keeps the tag/line per-point flags.
    "livox_pc2": [
        ("x", "x", "float32"), ("y", "y", "float32"), ("z", "z", "float32"),
        ("intensity", "intensity", "float32"), ("time", "time", "float32"),
        ("tag", "tag", "uint8"), ("line", "line", "uint8"),
    ],
}

def default_time_unit(type_name_: str) -> str:
    """Unit a per-point time column is written in, given its output type.

    The canonical `time` is float seconds, but an integer output column cannot
    hold that -- 0.1 s truncates to 0. Integer columns therefore default to
    nanoseconds (the Ouster `t` convention) and float columns to seconds (the
    Velodyne `time` convention). Deriving this from the *type* rather than from
    the preset name means a hand-written `fields:` list gets it right too; a
    config can still override with an explicit `unit:`.
    """
    return "s" if type_name_.startswith("float") else "ns"


def normalize_layout(spec: Any, preset_name: str = "") -> list[dict[str, Any]]:
    """Normalize a layout preset name or explicit field list to field dicts.

    An explicit list entry is ``{name, type, from?, offset?, unit?}``; ``from`` is
    the canonical source name (see CANONICAL_POINT_FIELDS) and defaults to
    ``name``, which is what lets a config rename a column (e.g. emit `t` instead
    of `time`) without touching any reader.
    """
    if isinstance(spec, str):
        if spec not in LAYOUT_PRESETS:
            raise SystemExit(
                f"unknown layout preset {spec!r}; known: {sorted(LAYOUT_PRESETS)}"
                f" (or give an explicit `fields:` list)"
            )
        return [{"name": n, "from": s, "type": t, "unit": default_time_unit(t)}
                for n, s, t in LAYOUT_PRESETS[spec]]
    if not isinstance(spec, list) or not spec:
        raise SystemExit(f"{preset_name or 'layout'} must be a preset name or a non-empty field list")
    out = []
    for f in spec:
        name = f["name"]
        typ = f.get("type", "float32")
        if typ not in NAME_TO_DATATYPE:
            raise SystemExit(
                f"unknown point field type {typ!r} for field {name!r}; "
                f"known: {sorted(NAME_TO_DATATYPE)}"
            )
        unit = f.get("unit", default_time_unit(typ))
        if unit not in TIME_UNIT_TO_S:
            raise SystemExit(f"unknown unit {unit!r} for field {name!r}; "
                             f"known: {sorted(TIME_UNIT_TO_S)}")
        entry = {"name": name, "from": f.get("from", name), "type": typ, "unit": unit}
        if "offset" in f:
            entry["offset"] = int(f["offset"])
        out.append(entry)
    return out


def build_layout(fields: list[dict[str, Any]], point_step: int | None = None
                 ) -> tuple[np.dtype, list[dict[str, Any]], int]:
    """Assign offsets, build the packed numpy dtype, return (dtype, fields, step).

    Offsets default to tight packing in declaration order; any field may pin its
    own. ``point_step`` defaults to the end of the last field -- note this reads
    the *layout-level* key, not a key buried in the first field, which is where
    the previous implementation looked for it (and so never found it).
    """
    cursor, resolved = 0, []
    for f in fields:
        code, _ = PC2_DATATYPES[NAME_TO_DATATYPE[f["type"]]]
        offset = f.get("offset", cursor)
        resolved.append({**f, "offset": offset})
        cursor = offset + np.dtype(code).itemsize
    step = int(point_step) if point_step else cursor
    if step < cursor:
        raise SystemExit(f"point_step {step} is smaller than the packed field extent {cursor}")
    dtype = np.dtype({
        "names": [f["name"] for f in resolved],
        "formats": ["<" + PC2_DATATYPES[NAME_TO_DATATYPE[f["type"]]][0]
                    if not PC2_DATATYPES[NAME_TO_DATATYPE[f["type"]]][0].endswith("1")
                    else PC2_DATATYPES[NAME_TO_DATATYPE[f["type"]]][0] for f in resolved],
        "offsets": [f["offset"] for f in resolved],
        "itemsize": step,
    })
    return dtype, resolved, step
