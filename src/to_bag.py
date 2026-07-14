#!/usr/bin/env python3
"""
Config-driven converter from dataset folders to ROS 2 bags.

The converter keeps dataset-specific choices in YAML:
  - where LiDAR/IMU/GT files live
  - timestamp units and offsets
  - LiDAR input format (.bin Livox CustomPoint or PCD XYZI)
  - output topics and message types
"""
import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy.serialization as ser
import rosbag2_py
import yaml
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage


LIVOX_BIN_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("refl", "u1"), ("tag", "u1"), ("line", "u1"),
    ("offset", "<u4"),
])

POINT_FIELD_TYPES = {
    "float32": PointField.FLOAT32,
    "uint8": PointField.UINT8,
}

NUMPY_TYPES = {
    "float32": "<f4",
    "uint8": "u1",
}

TIME_SCALE_TO_NS = {
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "us": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
}


@dataclass
class LidarFrame:
    path: str
    stamp_ns: int


def ns_to_time(ns: int) -> TimeMsg:
    return TimeMsg(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def time_value_to_ns(value: Any, unit: str) -> int:
    scale = TIME_SCALE_TO_NS[unit]
    return int(round(float(value) * scale))


def time_shift_to_ns(cfg: dict[str, Any]) -> int:
    if "time_shift_ns" in cfg:
        return int(cfg["time_shift_ns"])
    if "time_shift_sec" in cfg:
        return int(round(float(cfg["time_shift_sec"]) * 1e9))
    return 0


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(seq_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.join(seq_dir, value)


def stamp_from_filename(path: str, unit: str, shift_ns: int) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    return time_value_to_ns(stem, unit) + shift_ns


def list_lidar_frames(seq_dir: str, lidar_cfg: dict[str, Any]) -> list[LidarFrame]:
    data_dir = resolve_path(seq_dir, lidar_cfg["dir"])
    pattern = lidar_cfg.get("pattern", "*")
    unit = lidar_cfg.get("timestamp_unit", "ns")
    shift_ns = time_shift_to_ns(lidar_cfg)
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    return [LidarFrame(path=p, stamp_ns=stamp_from_filename(p, unit, shift_ns)) for p in paths]


def rot_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
            0.25 * s,
        )
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        return (
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[2, 1] - r[1, 2]) / s,
        )
    if r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        return (
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
            (r[0, 2] - r[2, 0]) / s,
        )
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
    return (
        (r[0, 2] + r[2, 0]) / s,
        (r[1, 2] + r[2, 1]) / s,
        0.25 * s,
        (r[1, 0] - r[0, 1]) / s,
    )


def make_topic_metadata(idx: int, name: str, typ: str):
    try:
        return rosbag2_py.TopicMetadata(id=idx, name=name, type=typ, serialization_format="cdr")
    except TypeError:
        return rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr")


def load_csv(path: str, delimiter: str = ",") -> np.ndarray:
    data = np.loadtxt(path, delimiter=delimiter)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_imu(seq_dir: str, imu_cfg: dict[str, Any]) -> np.ndarray:
    return load_csv(resolve_path(seq_dir, imu_cfg["path"]), imu_cfg.get("delimiter", ","))


def build_imu(row: np.ndarray, frame_id: str, time_unit: str) -> tuple[int, Imu]:
    stamp_ns = time_value_to_ns(row[0], time_unit)
    msg = Imu()
    msg.header = Header(stamp=ns_to_time(stamp_ns), frame_id=frame_id)
    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = row[1:5]
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = row[5:8]
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = row[8:11]
    return stamp_ns, msg


def pointcloud_dtype(fields_cfg: list[dict[str, Any]]) -> np.dtype:
    names = [f["name"] for f in fields_cfg]
    formats = [NUMPY_TYPES[f["type"]] for f in fields_cfg]
    offsets = [int(f["offset"]) for f in fields_cfg]
    itemsize = int(max(o + np.dtype(fmt).itemsize for o, fmt in zip(offsets, formats)))
    itemsize = int(fields_cfg[0].get("point_step", itemsize))
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": itemsize})


def pointcloud_fields(fields_cfg: list[dict[str, Any]]) -> list[PointField]:
    return [
        PointField(
            name=f["name"],
            offset=int(f["offset"]),
            datatype=POINT_FIELD_TYPES[f["type"]],
            count=int(f.get("count", 1)),
        )
        for f in fields_cfg
    ]


def configured_fields(output_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    fields = output_cfg.get("fields")
    if fields:
        return fields
    preset = output_cfg.get("fields_preset", "xyzi_t")
    if preset == "xyzi":
        return [
            {"name": "x", "type": "float32", "offset": 0},
            {"name": "y", "type": "float32", "offset": 4},
            {"name": "z", "type": "float32", "offset": 8},
            {"name": "intensity", "type": "float32", "offset": 12, "point_step": 16},
        ]
    if preset == "xyzi_t":
        return [
            {"name": "x", "type": "float32", "offset": 0},
            {"name": "y", "type": "float32", "offset": 4},
            {"name": "z", "type": "float32", "offset": 8},
            {"name": "intensity", "type": "float32", "offset": 12},
            {"name": "t", "type": "float32", "offset": 16, "point_step": 20},
        ]
    if preset == "xyzi_t_tag_line":
        return [
            {"name": "x", "type": "float32", "offset": 0},
            {"name": "y", "type": "float32", "offset": 4},
            {"name": "z", "type": "float32", "offset": 8},
            {"name": "intensity", "type": "float32", "offset": 12},
            {"name": "t", "type": "float32", "offset": 16},
            {"name": "tag", "type": "uint8", "offset": 20},
            {"name": "line", "type": "uint8", "offset": 21, "point_step": 24},
        ]
    raise ValueError(f"unknown pointcloud fields_preset: {preset}")


def load_conpr_livox_bin(path: str, drop_zero_points: bool) -> np.ndarray:
    pts = np.fromfile(path, dtype=LIVOX_BIN_DTYPE)
    if drop_zero_points:
        valid = ~((pts["x"] == 0) & (pts["y"] == 0) & (pts["z"] == 0))
        pts = pts[valid]
    return pts


def parse_pcd(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD header missing DATA line: {path}")
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
            raise ValueError(f"PCD fields with COUNT != 1 are not supported yet: {path}")
        dtype_fields = []
        for name, size, typ in zip(fields, sizes, types):
            if typ == "F" and size == 4:
                fmt = "<f4"
            elif typ == "F" and size == 8:
                fmt = "<f8"
            elif typ == "U" and size == 1:
                fmt = "u1"
            elif typ == "U" and size == 2:
                fmt = "<u2"
            elif typ == "U" and size == 4:
                fmt = "<u4"
            elif typ == "I" and size == 1:
                fmt = "i1"
            elif typ == "I" and size == 2:
                fmt = "<i2"
            elif typ == "I" and size == 4:
                fmt = "<i4"
            else:
                raise ValueError(f"unsupported PCD field type {typ}{size} in {path}")
            dtype_fields.append((name, fmt))
        dtype = np.dtype(dtype_fields)
        if data_kind == "ascii":
            arr = np.loadtxt(f, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return {name: arr[:, i] for i, name in enumerate(fields)}
        if data_kind == "binary":
            arr = np.frombuffer(f.read(dtype.itemsize * points), dtype=dtype, count=points)
            return {name: arr[name] for name in fields}
        raise ValueError(f"unsupported PCD DATA mode {data_kind}: {path}")


def pcd_to_xyzi(path: str, drop_zero_points: bool) -> np.ndarray:
    data = parse_pcd(path)
    n = len(data["x"])
    out = np.empty(n, dtype={
        "names": ["x", "y", "z", "intensity"],
        "formats": ["<f4", "<f4", "<f4", "<f4"],
        "offsets": [0, 4, 8, 12],
        "itemsize": 16,
    })
    out["x"] = data["x"].astype(np.float32)
    out["y"] = data["y"].astype(np.float32)
    out["z"] = data["z"].astype(np.float32)
    intensity = data.get("intensity", data.get("i"))
    out["intensity"] = 0.0 if intensity is None else intensity.astype(np.float32)
    if drop_zero_points:
        valid = ~((out["x"] == 0) & (out["y"] == 0) & (out["z"] == 0))
        out = out[valid]
    return out


def load_lidar_points(path: str, lidar_cfg: dict[str, Any]) -> np.ndarray:
    drop_zero_points = bool(lidar_cfg.get("drop_zero_points", True))
    fmt = lidar_cfg["format"]
    if fmt == "conpr_livox_bin":
        return load_conpr_livox_bin(path, drop_zero_points)
    if fmt == "pcd_xyzi":
        return pcd_to_xyzi(path, drop_zero_points)
    raise ValueError(f"unknown lidar format: {fmt}")


def map_points_to_output(raw: np.ndarray, lidar_cfg: dict[str, Any], output_cfg: dict[str, Any]) -> np.ndarray:
    fields_cfg = configured_fields(output_cfg)
    out_dtype = pointcloud_dtype(fields_cfg)
    out = np.zeros(raw.shape[0], dtype=out_dtype)
    fmt = lidar_cfg["format"]
    if fmt == "conpr_livox_bin":
        out["x"] = raw["x"]
        out["y"] = raw["y"]
        out["z"] = raw["z"]
        if "intensity" in out.dtype.names:
            out["intensity"] = raw["refl"].astype(np.float32)
        if "t" in out.dtype.names:
            out["t"] = raw["offset"].astype(np.float32) * 1e-9
        if "tag" in out.dtype.names:
            out["tag"] = raw["tag"]
        if "line" in out.dtype.names:
            out["line"] = raw["line"]
        return out
    if fmt == "pcd_xyzi":
        out["x"] = raw["x"]
        out["y"] = raw["y"]
        out["z"] = raw["z"]
        if "intensity" in out.dtype.names:
            out["intensity"] = raw["intensity"]
        return out
    raise ValueError(f"unknown lidar format: {fmt}")


def build_pointcloud2(points: np.ndarray, stamp_ns: int, frame_id: str, fields_cfg: list[dict[str, Any]]) -> PointCloud2:
    msg = PointCloud2()
    msg.header = Header(stamp=ns_to_time(stamp_ns), frame_id=frame_id)
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = pointcloud_fields(fields_cfg)
    msg.is_bigendian = False
    msg.point_step = points.dtype.itemsize
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True
    msg.data = points.tobytes()
    return msg


def build_livox_custom_msg(raw: np.ndarray, stamp_ns: int, frame_id: str, output_cfg: dict[str, Any]):
    try:
        from livox_ros_driver2.msg import CustomMsg, CustomPoint
    except ImportError as exc:
        raise RuntimeError("livox_ros_driver2 must be sourced to write Livox CustomMsg bags") from exc
    scale = output_cfg.get("offset_time_scale", "ns_to_us")
    msg = CustomMsg()
    msg.header = Header(stamp=ns_to_time(stamp_ns), frame_id=frame_id)
    msg.timebase = int(stamp_ns)
    msg.point_num = int(raw.shape[0])
    msg.lidar_id = int(output_cfg.get("lidar_id", 0))
    msg.rsvd = list(output_cfg.get("rsvd", [0, 0, 0]))
    points = []
    for point in raw:
        offset = int(point["offset"])
        if scale == "ns_to_us":
            offset = int(round(offset / 1000.0))
        elif scale != "identity":
            raise ValueError(f"unknown Livox offset_time_scale: {scale}")
        points.append(CustomPoint(
            offset_time=offset,
            x=float(point["x"]),
            y=float(point["y"]),
            z=float(point["z"]),
            reflectivity=int(point["refl"]),
            tag=int(point["tag"]),
            line=int(point["line"]),
        ))
    msg.points = points
    return msg


def build_lidar_msg(path: str, stamp_ns: int, cfg: dict[str, Any]):
    lidar_cfg = cfg["dataset"]["lidar"]
    output_cfg = cfg["bag"]["lidar"]
    frame_id = cfg["frames"]["lidar"]
    raw = load_lidar_points(path, lidar_cfg)
    message = output_cfg.get("message", "pointcloud2")
    if message == "pointcloud2":
        fields_cfg = configured_fields(output_cfg)
        points = map_points_to_output(raw, lidar_cfg, output_cfg)
        return build_pointcloud2(points, stamp_ns, frame_id, fields_cfg)
    if message == "livox_custom":
        if lidar_cfg["format"] != "conpr_livox_bin":
            raise ValueError("Livox CustomMsg output requires conpr_livox_bin input")
        return build_livox_custom_msg(raw, stamp_ns, frame_id, output_cfg)
    raise ValueError(f"unknown lidar output message: {message}")


def load_gt(seq_dir: str, gt_cfg: dict[str, Any]) -> np.ndarray | None:
    if not gt_cfg.get("enabled", False):
        return None
    return load_csv(resolve_path(seq_dir, gt_cfg["path"]), gt_cfg.get("delimiter", ","))


def build_odom(stamp_ns: int, row: np.ndarray, quat: np.ndarray, map_frame: str, child_frame: str) -> Odometry:
    msg = Odometry()
    msg.header = Header(stamp=ns_to_time(stamp_ns), frame_id=map_frame)
    msg.child_frame_id = child_frame
    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = row[1:4]
    msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = quat
    return msg


def nearest_row(rows: np.ndarray, stamps_ns: np.ndarray, stamp_ns: int) -> tuple[int, np.ndarray]:
    idx = int(np.abs(stamps_ns - stamp_ns).argmin())
    return idx, rows[idx]


def build_tf_static(stamp_ns: int, cfg: dict[str, Any]) -> TFMessage:
    tf_cfg = cfg["bag"].get("tf_static", {})
    matrix = np.array(tf_cfg["matrix"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("bag.tf_static.matrix must be 4x4")
    if tf_cfg.get("matrix_is", "parent_to_child") == "child_to_parent":
        matrix = np.linalg.inv(matrix)
    qx, qy, qz, qw = rot_to_quat(matrix[:3, :3])
    msg = TransformStamped()
    msg.header = Header(stamp=ns_to_time(stamp_ns), frame_id=tf_cfg.get("parent_frame", cfg["frames"]["imu"]))
    msg.child_frame_id = tf_cfg.get("child_frame", cfg["frames"]["lidar"])
    msg.transform.translation.x = float(matrix[0, 3])
    msg.transform.translation.y = float(matrix[1, 3])
    msg.transform.translation.z = float(matrix[2, 3])
    msg.transform.rotation.x = qx
    msg.transform.rotation.y = qy
    msg.transform.rotation.z = qz
    msg.transform.rotation.w = qw
    return TFMessage(transforms=[msg])


def output_topics(cfg: dict[str, Any], gt: np.ndarray | None) -> list[tuple[str, str]]:
    topics = []
    lidar_cfg = cfg["bag"]["lidar"]
    if lidar_cfg.get("enabled", True):
        msg = lidar_cfg.get("message", "pointcloud2")
        typ = "sensor_msgs/msg/PointCloud2" if msg == "pointcloud2" else "livox_ros_driver2/msg/CustomMsg"
        topics.append((lidar_cfg["topic"], typ))
    imu_cfg = cfg["bag"]["imu"]
    if imu_cfg.get("enabled", True):
        topics.append((imu_cfg["topic"], "sensor_msgs/msg/Imu"))
    gt_cfg = cfg["bag"].get("gt", {})
    if gt is not None and gt_cfg.get("enabled", False):
        topics.append((gt_cfg["topic"], "nav_msgs/msg/Odometry"))
    tf_cfg = cfg["bag"].get("tf_static", {})
    if tf_cfg.get("enabled", False):
        topics.append((tf_cfg.get("topic", "/tf_static"), "tf2_msgs/msg/TFMessage"))
    return topics


def convert(seq_dir: str, out: str, cfg: dict[str, Any]) -> None:
    lidar_frames = list_lidar_frames(seq_dir, cfg["dataset"]["lidar"])
    imu = load_imu(seq_dir, cfg["dataset"]["imu"])
    imu_time_unit = cfg["dataset"]["imu"].get("timestamp_unit", "ns")
    imu_stamps_ns = np.array([time_value_to_ns(row[0], imu_time_unit) for row in imu], dtype=np.int64)
    gt = load_gt(seq_dir, cfg["dataset"].get("gt", {}))
    gt_cfg = cfg["dataset"].get("gt", {})
    gt_stamps_ns = None
    if gt is not None:
        gt_unit = gt_cfg.get("timestamp_unit", "ns")
        gt_stamps_ns = np.array([time_value_to_ns(row[0], gt_unit) for row in gt], dtype=np.int64)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=out, storage_id=cfg["bag"].get("storage_id", "sqlite3")),
        rosbag2_py.ConverterOptions("", ""),
    )
    for idx, (name, typ) in enumerate(output_topics(cfg, gt)):
        writer.create_topic(make_topic_metadata(idx, name, typ))

    events = [(int(t), "imu", i) for i, t in enumerate(imu_stamps_ns)]
    events += [(frame.stamp_ns, "lidar", i) for i, frame in enumerate(lidar_frames)]
    gt_bag_cfg = cfg["bag"].get("gt", {})
    if gt is not None and gt_bag_cfg.get("enabled", False):
        publish_at = gt_bag_cfg.get("publish_at", "lidar")
        if publish_at == "lidar":
            events += [(frame.stamp_ns, "gt", i) for i, frame in enumerate(lidar_frames)]
        elif publish_at == "gt":
            events += [(int(t), "gt", i) for i, t in enumerate(gt_stamps_ns)]
        else:
            raise ValueError(f"unknown GT publish_at: {publish_at}")
    events.sort(key=lambda e: (e[0], e[1]))

    start_ns = events[0][0] if events else 0
    tf_cfg = cfg["bag"].get("tf_static", {})
    if tf_cfg.get("enabled", False):
        writer.write(tf_cfg.get("topic", "/tf_static"), ser.serialize_message(build_tf_static(start_ns, cfg)), start_ns)

    counts = {"lidar": 0, "imu": 0, "gt": 0}
    for stamp_ns, kind, idx in events:
        if kind == "imu":
            _, msg = build_imu(imu[idx], cfg["frames"]["imu"], imu_time_unit)
            writer.write(cfg["bag"]["imu"]["topic"], ser.serialize_message(msg), stamp_ns)
            counts["imu"] += 1
        elif kind == "lidar":
            frame = lidar_frames[idx]
            msg = build_lidar_msg(frame.path, stamp_ns, cfg)
            writer.write(cfg["bag"]["lidar"]["topic"], ser.serialize_message(msg), stamp_ns)
            counts["lidar"] += 1
        elif kind == "gt":
            if gt_bag_cfg.get("publish_at", "lidar") == "lidar":
                assert gt_stamps_ns is not None
                _, pose_row = nearest_row(gt, gt_stamps_ns, stamp_ns)
            else:
                pose_row = gt[idx]
            source = gt_bag_cfg.get("orientation", "row")
            if source == "row":
                quat = pose_row[4:8]
            elif source == "nearest_imu":
                _, imu_row = nearest_row(imu, imu_stamps_ns, stamp_ns)
                quat = imu_row[1:5]
            else:
                raise ValueError(f"unknown GT orientation source: {source}")
            msg = build_odom(stamp_ns, pose_row, quat, cfg["frames"]["map"], cfg["frames"]["imu"])
            writer.write(gt_bag_cfg["topic"], ser.serialize_message(msg), stamp_ns)
            counts["gt"] += 1

    pieces = [f"{counts['lidar']} lidar", f"{counts['imu']} imu"]
    if gt is not None and gt_bag_cfg.get("enabled", False):
        pieces.append(f"{counts['gt']} gt")
    print(f"done: {', '.join(pieces)} -> {out}")


def main(argv: list[str] | None = None, default_config: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_dir", help="sequence directory")
    parser.add_argument("-o", "-out", "--out", "--output", dest="output", required=True, help="output bag directory")
    parser.add_argument(
        "-c",
        "--config",
        default=default_config,
        required=default_config is None,
        help="bag conversion YAML config",
    )
    args = parser.parse_args(argv)
    convert(args.seq_dir, args.output, load_yaml(args.config))


if __name__ == "__main__":
    main()
