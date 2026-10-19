#!/usr/bin/env python3
"""Log /localization_status + /pose_with_covariance to CSV for one localizer run.

    monitor.py -o pose.csv [--seconds 600]

One row per status message. The pose columns hold the latest /pose_with_covariance with its
own header stamp (`pose_stamp_s`), which is the instant the pose describes; `stamp_s` is the
status message's stamp.

The columns are those of src/hilti/monitor_pose.py, in the same order, followed by the
alignment diagnostics kd_amr_libs main added to LocalizationStatus (align_match_count ...
prior_align_ratio). A field the running localizer version does not publish is written empty,
so one schema serves every version and a reader can tell "absent" from "zero".
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
COLUMNS = ("wall_s,stamp_s,status,lost_reason,ray_inlier_ratio,sigma_factor,"
           "pose_stamp_s,x,y,z,qx,qy,qz,qw,"
           "sig_pos_x,sig_pos_y,sig_pos_z,sig_rot_x,sig_rot_y,sig_rot_z,"
           "odometry_mode,odometry_distance,"
           "align_match_count,align_match_ratio,align_residual_rms,align_iterations,align_converged,"
           "prior_align_ratio\n")


def optional(msg, field, fmt):
    """`field` formatted with `fmt`, or "" when this LocalizationStatus version lacks it."""
    value = getattr(msg, field, None)
    return "" if value is None else format(value, fmt)


def optional_flag(msg, field):
    value = getattr(msg, field, None)
    return "" if value is None else str(int(value))


class Monitor(Node):
    def __init__(self, out):
        super().__init__("localization_monitor")
        self.out, self.t0, self.rows = out, time.time(), 0
        self.status = self.pose = None
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(LocalizationStatus, "/localization_status", self.on_status, qos)
        self.create_subscription(PoseWithCovarianceStamped, "/pose_with_covariance", self.on_pose, 50)
        out.write(COLUMNS)

    def on_status(self, msg):
        self.status = msg
        self.write_row()

    def on_pose(self, msg):
        self.pose = msg

    def write_row(self):
        s = self.status
        stamp = s.header.stamp.sec + s.header.stamp.nanosec * 1e-9
        row = [f"{time.time() - self.t0:.2f}", f"{stamp:.3f}", LABELS.get(s.status, "?"),
               '"' + s.lost_reason.replace('"', "'") + '"',
               optional(s, "ray_inlier_ratio", ".4f"), f"{s.pose_sigma_factor_increase:.3f}"]
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
        row += [optional_flag(s, "odometry_mode"), optional(s, "odometry_distance", ".3f")]
        row += [optional(s, "align_match_count", "d"), optional(s, "align_match_ratio", ".4f"),
                optional(s, "align_residual_rms", ".5f"), optional(s, "align_iterations", "d"),
                optional_flag(s, "align_converged"), optional(s, "prior_align_ratio", ".4f")]
        self.out.write(",".join(row) + "\n")
        self.out.flush()
        self.rows += 1
        if self.rows % 100 == 0:
            print(f"  {self.rows} status messages, last {LABELS.get(s.status)}, "
                  f"odometry mode {optional_flag(s, 'odometry_mode') or '-'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--seconds", type=float, default=900.0, help="stop after this many seconds")
    args = parser.parse_args()
    rclpy.init()
    with open(args.output, "w") as out:
        node = Monitor(out)
        end = time.time() + args.seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.5)
        print(f"wrote {node.rows} rows to {args.output}", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
