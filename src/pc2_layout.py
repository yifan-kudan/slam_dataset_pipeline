"""Shared PointCloud2 layout helpers for the bag tools.

Splits the PointCloud2 handling into the part that is **fixed by the ROS message
spec** and the part that **varies per dataset**, so the varying part can be
configured instead of edited:

fixed (lives here as constants, never needs touching)
  * ``PC2_DATATYPES`` -- the ``sensor_msgs/PointField.datatype`` enum. Values
    1..8 (INT8..FLOAT64), identical in ROS 1 and ROS 2, so any spec-conforming
    cloud is covered.
  * byte order -- taken from each message's own ``is_bigendian`` flag rather than
    hardcoded little-endian.

varies per dataset (configurable; defaults below are just the common cases)
  * field *names*: the per-point time field is ``time`` on Velodyne, ``t`` on
    Ouster, ``timestamp`` on Hesai, ``offset_time`` on Livox; the ring field is
    ``ring``, ``channel``, ``line`` or ``laser_id``. See ``DEFAULT_TIME_FIELDS`` /
    ``DEFAULT_RING_FIELDS`` -- ``inspect_bag.py`` takes overrides on the command
    line, ``rosbag1_to_ros2.py`` takes the chosen name as ``time_field``.
  * field *units*: the same ``float32 time`` may hold seconds, milliseconds,
    microseconds or nanoseconds. See ``TIME_UNIT_TO_NS``; the converter takes
    ``time_unit`` per topic and ``inspect_bag.py`` guesses it from the value range.
  * absolute vs scan-relative per-point time (Hesai ships absolute Unix seconds).
    ``epoch_kind`` classifies a stamp; the converter's ``repack`` handles it.

Used by ``inspect_bag.py`` (read-only probing) and ``rosbag1_to_ros2.py``
(rewriting), which previously carried two near-duplicate copies of the datatype
table and the strided-read idiom.
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

# Per-point time / ring field names, by vendor. Only defaults: pass your own when
# a dataset uses something else.
DEFAULT_TIME_FIELDS: tuple[str, ...] = (
    "time",         # Velodyne / FAST-LIO velodyne_handler
    "t",            # Ouster
    "timestamp",    # Hesai (PandarXT), often absolute Unix seconds
    "time_offset",
    "offset_time",  # Livox
)
DEFAULT_RING_FIELDS: tuple[str, ...] = ("ring", "channel", "line", "laser_id")

# Per-point time units seen in the wild -> multiplier to nanoseconds / to seconds.
# Ordered coarse->fine; guess_time_unit() relies on that order. Both tables are
# spelled out rather than deriving one from the other: absolute Unix values are
# ~1.7e9 s, and routing them through nanoseconds (*1e9 then /1e9) costs ~130 ns of
# float64 precision -- visible once the result lands in a float32 `time` field.
TIME_UNIT_TO_NS: dict[str, float] = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
TIME_UNIT_TO_S: dict[str, float] = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}

# One LiDAR revolution at 5-20 Hz is 0.05-0.2 s. Allow an order of magnitude
# either side before calling an observed span implausible for a scan.
SCAN_SPAN_MIN_S, SCAN_SPAN_MAX_S = 0.005, 2.0

# Plausible-Unix-time window (ns): 2001-09-09 .. 2033-05-18.
UNIX_LO_NS, UNIX_HI_NS = 1_000_000_000_000_000_000, 2_000_000_000_000_000_000


def type_name(datatype: int) -> str:
    """Human-readable name for a PointField datatype enum value."""
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
    """First PointField in ``msg`` whose name is in ``candidates`` (in that order)."""
    by_name = {f.name: f for f in msg.fields}
    for name in candidates:
        if name in by_name:
            return by_name[name]
    return None


def field_view(buf: Any, msg: Any, field: Any, strict: bool = False) -> np.ndarray:
    """Zero-copy strided view of one field across every point of ``msg``.

    PointCloud2 stores points interleaved, so field *i* of point *k* lives at
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


def guess_time_unit(span: float) -> str | None:
    """Guess a per-point time field's unit from its observed min..max span.

    The span of one scan is one revolution, so whichever unit maps the raw span
    into a plausible scan duration is almost certainly the right one. Returns
    None when no unit fits (all-zero field, or something that is not a scan).
    """
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
        better = guess_time_unit(span_ns / TIME_UNIT_TO_NS[unit])
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
