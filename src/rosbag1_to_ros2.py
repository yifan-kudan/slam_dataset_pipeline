"""
Config-driven converter from a ROS 1 ``.bag`` to a ROS 2 bag.

Unlike ``to_bag.py`` (which builds a bag from LiDAR files + IMU CSV), 
"""
import argparse
import os
from typing import Any

import numpy as np
import yaml
from rosbags.rosbag1 import Reader
from rosbags.rosbag2 import Writer
from rosbags.rosbag2.enums import StoragePlugin
from rosbags.typesys import Stores, get_typestore
from rosbags.interfaces import (
    Qos, QosDurability, QosHistory, QosLiveliness, QosReliability, QosTime,
)

# PointField datatype table, endian-aware strided reads, and the per-dataset
# knobs (field names, per-point time units). Shared with inspect_bag.py.
from pc2_layout import (
    epoch_kind, field_view, find_field, span_warning, to_ns, to_seconds,
)

ROS1_STORE = get_typestore(Stores.ROS1_NOETIC) # Setup ROS 1 type store for deserialization
ROS2_STORE = get_typestore(Stores.ROS2_JAZZY) # Setup ROS 2 type store for serialization

STORAGE_PLUGINS = { # Two storage plugins are supported: SQLite3 and MCAP.
    "sqlite3": StoragePlugin.SQLITE3,
    "mcap": StoragePlugin.MCAP,
}

# Topics whose per-point time span already drew a warning -- warn once per topic
# rather than once per cloud.
_warned_topics: set[str] = set()


def warn_once(topic: str, message: str | None) -> None:
    if message and topic not in _warned_topics:
        _warned_topics.add(topic)
        print(f"WARNING [{topic}]: {message}")


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_topic(name: str) -> str:
    return name if name.startswith("/") else "/" + name

# shift the per-point time field so it starts at 0, and adjust the header stamp
def rebase_pointcloud_time(msg, time_field: str, time_unit: str = "s",
                           topic: str = "") -> tuple[Any, int]:
    """Rebase the per-point time column to start at 0; return the header shift.

    ``time_unit`` is the unit the field is stored in ("s", "ms", "us", "ns") --
    dataset dependent, so it comes from the topic config. Getting it wrong would
    silently scale every stamp by a power of 1000, hence the span check.
    """
    field = find_field(msg, [time_field])
    if field is None:
        return msg, 0
    buf = bytearray(msg.data)  # mutable: the column is rewritten in place
    times = field_view(buf, msg, field, strict=True)
    tmin, tmax = float(times.min()), float(times.max())
    warn_once(topic or time_field, span_warning(to_ns(tmax - tmin, time_unit),
                                                field.name, time_unit))
    if tmin != 0.0:
        times[:] = times - np.asarray(tmin, dtype=times.dtype)
    msg.data = np.frombuffer(buf, dtype=np.uint8)
    stamp_delta_ns = int(round(to_ns(tmin, time_unit)))
    return msg, stamp_delta_ns


def advance_stamp(stamp, delta_ns: int):
    total = stamp.sec * 1_000_000_000 + stamp.nanosec + delta_ns
    stamp.sec = int(total // 1_000_000_000)
    stamp.nanosec = int(total % 1_000_000_000)
    return stamp


def repack_hesai_to_velodyne(msg, time_field: str = "timestamp", time_unit: str = "s",
                             ring_field: str = "ring", topic: str = ""):
    """Rewrite a Hesai PandarXT cloud into the Velodyne layout FAST-LIO reads.

    Hesai (Hilti) publishes ``x,y,z,intensity`` (float32), ``timestamp``
    (float64, ABSOLUTE Unix seconds -- ``relative_point_timestamp: false`` in
    the official calibration) and ``ring`` (uint16), point_step 48. FAST-LIO's
    ``velodyne_handler`` does ``pcl::fromROSMsg`` into ``velodyne_ros::Point``,
    which is registered as ``x,y,z,intensity,time`` (float32) ``,ring``
    (uint16): it matches fields *by name*, so it needs a float32 field literally
    called ``time`` holding the per-point offset from the scan start, not an
    absolute float64 ``timestamp``. Rebuild the point buffer accordingly:
    ``time = timestamp - header_stamp`` in SECONDS (use ``timestamp_unit: 0`` in
    the FAST-LIO config). The header stamp is left untouched -- it already marks
    the scan start, so LiDAR<->IMU sync is preserved.

    The *source* names and unit (``time_field``, ``time_unit``, ``ring_field``)
    are configurable because they vary by dataset; the *target* layout is fixed
    by FAST-LIO's point struct, so it stays hardcoded. The absolute-Unix-time
    assumption is checked rather than assumed -- feeding relative time here would
    subtract ~1.7e9 s from every point.
    """
    fmap = {f.name: f for f in msg.fields}
    missing = {"x", "y", "z", "intensity", time_field, ring_field} - set(fmap)
    if missing:
        raise SystemExit(f"hesai_to_velodyne: cloud is missing fields {sorted(missing)}; "
                         f"present: {sorted(fmap)}")
    n = msg.width * msg.height
    src = np.asarray(msg.data, dtype=np.uint8).tobytes()

    def col(name: str, np_type: str):
        f = fmap[name]
        return np.ndarray((n,), dtype=np_type, buffer=src,
                          offset=f.offset, strides=(msg.point_step,))

    # Endian-aware read of the source time column, then validate it really is
    # absolute Unix time before differencing against the header stamp.
    tsrc = field_view(src, msg, fmap[time_field]).astype("<f8")
    tmin, tmax = float(tsrc.min()), float(tsrc.max())
    kind = epoch_kind(to_ns(tmin, time_unit))
    if kind != "unix_absolute":
        raise SystemExit(
            f"hesai_to_velodyne: '{time_field}' looks {kind} (min={tmin:.6g} "
            f"{time_unit}), not absolute Unix time. This repack subtracts the "
            f"header stamp; for already-relative per-point time use "
            f"rebase_point_time instead."
        )
    warn_once(topic or time_field,
              span_warning(to_ns(tmax - tmin, time_unit), time_field, time_unit))

    header_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
    out = np.zeros(n, dtype=np.dtype({
        "names": ["x", "y", "z", "intensity", "time", "ring"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
        "offsets": [0, 4, 8, 12, 16, 20],
        "itemsize": 22,
    }))
    out["x"] = col("x", "<f4")
    out["y"] = col("y", "<f4")
    out["z"] = col("z", "<f4")
    out["intensity"] = col("intensity", "<f4")
    # FAST-LIO's `time` is seconds from scan start, whatever unit the source used.
    # to_seconds (not to_ns/1e9): a ns round-trip on ~1.7e9 s values loses ~130 ns.
    out["time"] = (to_seconds(tsrc, time_unit) - header_sec).astype("<f4")
    out["ring"] = col(ring_field, "<u2")

    # Reuse the deserialized PointField objects (they serialize back out with
    # ROS2_STORE just like the originals) rather than constructing new ones.
    x, y, z = fmap["x"], fmap["y"], fmap["z"]
    inten, tf, ring = fmap["intensity"], fmap[time_field], fmap[ring_field]
    inten.offset = 12
    tf.name, tf.datatype, tf.offset = "time", 7, 16   # 7 = FLOAT32
    ring.name, ring.datatype, ring.offset = "ring", 4, 20   # 4 = UINT16
    msg.fields = [x, y, z, inten, tf, ring]
    msg.point_step = 22
    msg.row_step = 22 * msg.width
    msg.is_bigendian = False   # the buffer above is written little-endian
    msg.data = out.view(np.uint8).reshape(-1)
    return msg


# rmw "infinite" duration sentinel used by ROS 2 QoS profiles.
_QOS_INFINITE = QosTime(sec=9223372036, nsec=854775807)

# /tf_static is published transient_local (latched) so late subscribers -- like
# kd_localizer coming up after bag playback starts -- still receive the one-shot
# extrinsics. Record the connection with that QoS so `ros2 bag play` re-offers it.
TF_STATIC_QOS = Qos(
    history=QosHistory.KEEP_LAST, depth=1,
    reliability=QosReliability.RELIABLE, durability=QosDurability.TRANSIENT_LOCAL,
    deadline=_QOS_INFINITE, lifespan=_QOS_INFINITE,
    liveliness=QosLiveliness.AUTOMATIC, liveliness_lease_duration=_QOS_INFINITE,
    avoid_ros_namespace_conventions=False,
)


def write_tf_static(writer: Writer, transforms: list[dict[str, Any]], stamp_ns: int) -> int:
    """Write a single latched /tf_static message from the config `tf_static` list.

    Each entry needs `parent`, `child`, `translation` [x,y,z] and `rotation_xyzw`
    [x,y,z,w]. kd_localizer reads base_frame_id->lidar (and ->imu when use_imu)
    from here; it is NOT needed by FAST-LIO (which takes extrinsics from its own
    YAML). All transforms are stamped at the bag start time so they precede the
    first sensor message on playback.
    """
    Time = ROS2_STORE.types["builtin_interfaces/msg/Time"]
    Header = ROS2_STORE.types["std_msgs/msg/Header"]
    Vector3 = ROS2_STORE.types["geometry_msgs/msg/Vector3"]
    Quaternion = ROS2_STORE.types["geometry_msgs/msg/Quaternion"]
    Transform = ROS2_STORE.types["geometry_msgs/msg/Transform"]
    TransformStamped = ROS2_STORE.types["geometry_msgs/msg/TransformStamped"]
    TFMessage = ROS2_STORE.types["tf2_msgs/msg/TFMessage"]

    stamp = Time(sec=int(stamp_ns // 1_000_000_000), nanosec=int(stamp_ns % 1_000_000_000))
    tfs = []
    for t in transforms:
        tx, ty, tz = t["translation"]
        qx, qy, qz, qw = t["rotation_xyzw"]
        tfs.append(TransformStamped(
            header=Header(stamp=stamp, frame_id=t["parent"]),
            child_frame_id=t["child"],
            transform=Transform(
                translation=Vector3(x=float(tx), y=float(ty), z=float(tz)),
                rotation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
            ),
        ))
    conn = writer.add_connection("/tf_static", "tf2_msgs/msg/TFMessage",
                                 typestore=ROS2_STORE, offered_qos_profiles=[TF_STATIC_QOS])
    writer.write(conn, stamp_ns,
                 ROS2_STORE.serialize_cdr(TFMessage(transforms=tfs), "tf2_msgs/msg/TFMessage"))
    return len(tfs)


def convert(in_bag: str, out_bag: str, cfg: dict[str, Any], limit_sec: float | None = None,
            start_sec: float | None = None) -> None:
    topic_cfgs = {t["in"]: t for t in cfg["topics"]}
    storage = cfg.get("output", {}).get("storage_id", "sqlite3")
    version = int(cfg.get("output", {}).get("version", 9))

    writer = Writer(out_bag, version=version, storage_plugin=STORAGE_PLUGINS[storage])
    writer.open()

    counts: dict[str, int] = {}
    with Reader(in_bag) as reader:
        conns = [c for c in reader.connections if c.topic in topic_cfgs]
        if not conns:
            raise SystemExit(
                f"none of the configured topics found in {in_bag}\n"
                f"  wanted: {sorted(topic_cfgs)}\n"
                f"  present: {sorted({c.topic for c in reader.connections})}"
            )
        out_conns = {}
        for c in conns:
            tc = topic_cfgs[c.topic]
            out_topic = normalize_topic(tc.get("out", c.topic))
            out_conns[c.topic] = writer.add_connection(out_topic, c.msgtype, typestore=ROS2_STORE)
            counts[out_topic] = 0

        t0 = reader.start_time
        tf_static = cfg.get("tf_static")
        if tf_static:
            n_tf = write_tf_static(writer, tf_static, t0)
            counts["/tf_static"] = n_tf
        last_kept_ns: dict[str, int] = {}
        last_out_ns: dict[str, int] = {}
        first_imu_out_ns: int | None = None
        for c, timestamp, raw in reader.messages(connections=conns):
            rel_sec = (timestamp - t0) / 1e9
            if start_sec is not None and rel_sec < start_sec:
                continue
            if limit_sec is not None and rel_sec > limit_sec:
                break
            tc = topic_cfgs[c.topic]

            # Drop messages that arrive closer than min_dt_sec to the previously
            # kept one on this topic (coarse decimation).
            min_dt_ns = int(round(float(tc.get("min_dt_sec", 0.0)) * 1e9))
            if min_dt_ns > 0:
                prev = last_kept_ns.get(c.topic)
                if prev is not None and (timestamp - prev) < min_dt_ns:
                    continue
                last_kept_ns[c.topic] = timestamp

            msg = ROS1_STORE.deserialize_ros1(raw, c.msgtype)

            frame_id = tc.get("frame_id")
            if frame_id is not None and hasattr(msg, "header"):
                msg.header.frame_id = frame_id

            write_stamp = timestamp
            # `time_field` / `time_unit` / `ring_field` are the dataset-dependent
            # parts of the cloud layout; everything else follows the ROS spec.
            time_unit = tc.get("time_unit", "s")
            if tc.get("type") == "pointcloud2" and tc.get("repack") == "hesai_to_velodyne":
                # Hesai per-point time is absolute Unix seconds -> repack to the
                # Velodyne layout with a relative float32 `time`. The header
                # stamp already marks scan start, so it (and write_stamp) stay.
                msg = repack_hesai_to_velodyne(
                    msg, tc.get("time_field", "timestamp"), time_unit,
                    tc.get("ring_field", "ring"), c.topic)
            elif tc.get("type") == "pointcloud2" and tc.get("rebase_point_time", False):
                msg, delta_ns = rebase_pointcloud_time(
                    msg, tc.get("time_field", "time"), time_unit, c.topic)
                advance_stamp(msg.header.stamp, delta_ns)
                write_stamp = timestamp + delta_ns

            # Delay the bag WRITE timestamp (leaving the header stamp untouched)
            # so a streaming consumer has the whole scan buffered before the frame
            # is delivered. kd_localizer consumes each LiDAR frame the instant
            # rosbag2 delivers it and immediately needs IMU spanning the full scan
            # for deskew + ESKF prediction; if the cloud is written at scan START
            # (header time), the ~one-scan of covering IMU arrives AFTER it and the
            # predictor reports "missing IMU samples" every frame. Shifting the
            # write stamp to scan END (> one scan period, e.g. 0.11 s for 10 Hz)
            # makes rosbag2 deliver that IMU first. The header stays at scan start
            # so deskew math -- and LiDAR<->IMU sync -- is unchanged. Absent from
            # the FAST-LIO config, which buffers a sliding window and wants
            # scan-start stamps.
            write_shift_ns = int(round(float(tc.get("write_stamp_shift_sec", 0.0)) * 1e9))
            if write_shift_ns != 0:
                write_stamp += write_shift_ns

            # Re-space clustered timestamps. ConSLAM's IMU arrives in bursts of
            # samples stamped within microseconds of each other (then a ~11 ms
            # gap); the near-zero intra-burst dt breaks FAST-LIO's IMU
            # integration. Push each stamp forward so consecutive samples are at
            # least respace_min_dt_sec apart, keeping every sample and drifting
            # back to real time whenever a real gap appears. Both the bag write
            # timestamp and the message header stamp are moved together.
            respace_ns = int(round(float(tc.get("respace_min_dt_sec", 0.0)) * 1e9))
            if respace_ns > 0:
                prev_out = last_out_ns.get(c.topic)
                spaced = write_stamp if prev_out is None else max(write_stamp, prev_out + respace_ns)
                last_out_ns[c.topic] = spaced
                if spaced != write_stamp and hasattr(msg, "header"):
                    advance_stamp(msg.header.stamp, spaced - write_stamp)
                write_stamp = spaced

            # Align LiDAR to IMU: drop leading LiDAR scans that precede the IMU
            # stream (plus a warmup margin), so the localizer's IMU predictor
            # always has samples bracketing the first scan it processes. ConSLAM's
            # IMU starts ~0.3 s after the LiDAR, which otherwise triggers
            # "missing IMU samples". Keyed on OUTPUT header stamps (post rebase /
            # respace) -- the sensor time the consumer actually reads.
            out_hdr_ns = None
            if hasattr(msg, "header"):
                out_hdr_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            if tc.get("type") == "imu" and out_hdr_ns is not None and first_imu_out_ns is None:
                first_imu_out_ns = out_hdr_ns
            warmup_sec = float(tc.get("warmup_after_imu_sec", 0.0))
            if tc.get("type") == "pointcloud2" and warmup_sec > 0.0 and out_hdr_ns is not None:
                if first_imu_out_ns is None or out_hdr_ns < first_imu_out_ns + int(warmup_sec * 1e9):
                    continue

            out_conn = out_conns[c.topic]
            writer.write(out_conn, write_stamp, ROS2_STORE.serialize_cdr(msg, c.msgtype))
            counts[out_conn.topic] += 1

    writer.close()
    summary = ", ".join(f"{n} {t}" for t, n in counts.items())
    print(f"done: {summary} -> {out_bag}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_bag", help="input ROS 1 .bag file")
    parser.add_argument("-o", "--out", "--output", dest="output", required=True,
                        help="output ROS 2 bag directory")
    parser.add_argument("-c", "--config", required=True, help="conversion YAML config")
    parser.add_argument("--limit-sec", type=float, default=None,
                        help="stop at N seconds from bag start (upper bound of the window)")
    parser.add_argument("--start-sec", type=float, default=None,
                        help="skip messages before N seconds from bag start (lower bound of the window)")
    args = parser.parse_args(argv)
    if args.start_sec is not None and args.limit_sec is not None and args.start_sec >= args.limit_sec:
        raise SystemExit(f"--start-sec ({args.start_sec}) must be < --limit-sec ({args.limit_sec})")
    if os.path.exists(args.output):
        raise SystemExit(f"output already exists: {args.output}")
    convert(args.in_bag, args.output, load_yaml(args.config), args.limit_sec, args.start_sec)


if __name__ == "__main__":
    main()
