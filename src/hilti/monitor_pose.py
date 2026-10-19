#!/usr/bin/env python3
"""Log /localization_status + /pose_with_covariance for a Hilti 2023 run.

Same job as kd_localizer_docker/tools/diag/monitor_localization.py, with two
additions this dataset
needs and the ConSLAM one did not:

* **orientation** (`qx qy qz qw`).  Hilti's ground truth is a surveyed prism, not
  the IMU origin -- 20 cm off-axis on the handheld rig and 0.77 m on the robot --
  so scoring a pose requires its attitude, not just its position.
* **the pose message's own stamp** (`pose_stamp_s`), separate from the status
  stamp the row is emitted on.  Interpolating to a control-point time is only as
  good as the timestamp attached to the pose being interpolated.

Works against both LocalizationStatus layouts. kd-localizer-v2.0.0 removed
`ray_inlier_ratio` and added `odometry_mode` / `odometry_distance`; whichever the
running node does not publish is written as an empty field, so one CSV schema
serves every version and readers can tell "absent" from "zero".

The original file is left alone: the ConSLAM artefacts are parsed by column
position and are still being read.

    ./monitor_pose.py -o out.csv [--seconds 600]
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from kd_amr_ros_interfaces.msg import LocalizationStatus
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

LABELS = {0: "UNINIT", 1: "TRACKING", 2: "LOST"}
COLS = ("wall_s,stamp_s,status,lost_reason,ray_inlier_ratio,sigma_factor,"
        "pose_stamp_s,x,y,z,qx,qy,qz,qw,"
        "sig_pos_x,sig_pos_y,sig_pos_z,sig_rot_x,sig_rot_y,sig_rot_z,"
        "odometry_mode,odometry_distance\n")


def _opt(msg, field, fmt):
    """`field` formatted, or "" when this LocalizationStatus version lacks it."""
    v = getattr(msg, field, None)
    return "" if v is None else format(v, fmt)


class Monitor(Node):
    def __init__(self, out):
        super().__init__("localization_monitor")
        self.out, self.t0, self.n = out, time.time(), 0
        self.status = self.pose = None
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(LocalizationStatus, "/localization_status", self.on_status, qos)
        self.create_subscription(
            PoseWithCovarianceStamped, "/pose_with_covariance", self.on_pose, 50)
        out.write(COLS)

    def on_status(self, msg):
        self.status = msg
        self.emit()

    def on_pose(self, msg):
        self.pose = msg

    def emit(self):
        s = self.status
        stamp = s.header.stamp.sec + s.header.stamp.nanosec * 1e-9
        row = [f"{time.time() - self.t0:.2f}", f"{stamp:.3f}", LABELS.get(s.status, "?"),
               '"' + s.lost_reason.replace('"', "'") + '"',
               _opt(s, "ray_inlier_ratio", ".4f"), f"{s.pose_sigma_factor_increase:.3f}"]
        if self.pose is not None:
            h = self.pose.header.stamp
            p = self.pose.pose.pose.position
            q = self.pose.pose.pose.orientation
            c = self.pose.pose.covariance
            row += [f"{h.sec + h.nanosec * 1e-9:.6f}"]
            row += [f"{p.x:.4f}", f"{p.y:.4f}", f"{p.z:.4f}"]
            row += [f"{q.x:.6f}", f"{q.y:.6f}", f"{q.z:.6f}", f"{q.w:.6f}"]
            row += [f"{math.sqrt(max(c[i * 6 + i], 0.0)):.6f}" for i in range(6)]
        else:
            row += [""] * 13
        odom = getattr(s, "odometry_mode", None)
        row += ["" if odom is None else str(int(odom)), _opt(s, "odometry_distance", ".3f")]
        self.out.write(",".join(row) + "\n")
        self.out.flush()
        self.n += 1
        if self.n % 100 == 0:
            print(f"  {self.n} status msgs, last {LABELS.get(s.status)} "
                  f"factor {s.pose_sigma_factor_increase:.2f} "
                  f"inlier {_opt(s, 'ray_inlier_ratio', '.3f') or '-'} "
                  f"odom {_opt(s, 'odometry_mode', 'd') or '-'}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seconds", type=float, default=900.0)
    a = ap.parse_args()
    rclpy.init()
    with open(a.output, "w") as out:
        node = Monitor(out)
        end = time.time() + a.seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.5)
        print(f"wrote {node.n} rows to {a.output}", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
