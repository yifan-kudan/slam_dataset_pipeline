#!/usr/bin/env python3
"""Read static extrinsics from a ROS 1/2 bag's ``/tf_static`` topic.

Some datasets (e.g. Hilti SLAM Challenge 2023) ship calibration not as YAML but
as a recorded bag: the sensor-to-sensor transforms are published on
``/tf_static`` (``tf2_msgs/TFMessage``) and recorded like any other topic. This
tool lists every static transform and, for a chosen ``parent -> child`` pair,
prints it in the FAST-LIO ``extrinsic_T`` / ``extrinsic_R`` layout (the child
sensor's pose expressed in the parent frame = FAST-LIO's LiDAR->IMU extrinsic
when parent=imu, child=lidar).

Reads with the pure-python ``rosbags`` library -- no ROS environment sourced.
IMPORTANT: use ``AnyReader`` (not the low-level ``Reader`` + a fixed typestore):
``tf2_msgs/msg/TFMessage`` is not in the built-in ROS1_NOETIC store, so a manual
``deserialize_ros1`` raises ``KeyError``. ``AnyReader`` registers each bag
connection's own embedded message definition, so it deserializes fine.

Example:
    python3 src/read_tf_extrinsic.py /home/kudan/Downloads/calibration_handheld.bag \
        --parent imu --child PandarXT-32
"""
import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def read_static_transforms(bag: Path) -> dict[tuple[str, str], tuple[float, ...]]:
    """Return {(parent, child): (x, y, z, qx, qy, qz, qw)} from /tf_static."""
    seen: dict[tuple[str, str], tuple[float, ...]] = {}
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == "/tf_static"]
        if not conns:
            raise SystemExit(f"no /tf_static topic in {bag}")
        for conn, _t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)  # AnyReader registers the bag's type defs
            for tr in msg.transforms:
                key = (tr.header.frame_id, tr.child_frame_id)
                if key not in seen:  # static TFs repeat every message; keep the first
                    p, q = tr.transform.translation, tr.transform.rotation
                    seen[key] = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    return seen


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path, help="ROS 1 .bag file or ROS 2 bag directory")
    ap.add_argument("--parent", help="parent frame_id of the transform to extract (e.g. imu)")
    ap.add_argument("--child", help="child_frame_id of the transform to extract (e.g. PandarXT-32)")
    args = ap.parse_args(argv)

    tfs = read_static_transforms(args.bag)

    print(f"# {len(tfs)} static transforms in {args.bag.name} /tf_static\n")
    for (parent, child), v in sorted(tfs.items()):
        x, y, z, qx, qy, qz, qw = v
        print(f"{parent:12s} -> {child:20s} "
              f"T=[{x:+.6f}, {y:+.6f}, {z:+.6f}]  q(xyzw)=[{qx:+.7f}, {qy:+.7f}, {qz:+.7f}, {qw:+.7f}]")

    if not (args.parent and args.child):
        print("\n# pass --parent/--child to print one transform as a FAST-LIO extrinsic")
        return

    key = (args.parent, args.child)
    if key not in tfs:
        raise SystemExit(f"\ntransform {args.parent} -> {args.child} not found")

    x, y, z, qx, qy, qz, qw = tfs[key]
    R = quat_to_matrix(qx, qy, qz, qw)
    print(f"\n# FAST-LIO extrinsic for {args.parent} -> {args.child} "
          f"({args.child} pose in the {args.parent} frame)")
    print(f"extrinsic_T: [{x:.6g}, {y:.6g}, {z:.6g}]")
    rows = [", ".join(f"{v:.6g}" for v in row) for row in R]
    print("extrinsic_R: [" + rows[0] + ",")
    print("              " + rows[1] + ",")
    print("              " + rows[2] + "]")


if __name__ == "__main__":
    main()
