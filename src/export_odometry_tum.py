#!/usr/bin/env python3
"""
Export a nav_msgs/Odometry topic from a ROS 2 bag to TUM trajectory text.

Output format:
  timestamp_sec x y z qx qy qz qw
"""
import argparse

import rclpy.serialization as ser
import rosbag2_py
from nav_msgs.msg import Odometry


def export_odometry(bag_path: str, topic: str, output_path: str) -> int:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=""),
        rosbag2_py.ConverterOptions("", ""),
    )

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        while reader.has_next():
            msg_topic, data, stamp_ns = reader.read_next()
            if msg_topic != topic:
                continue
            msg = ser.deserialize_message(data, Odometry)
            stamp = msg.header.stamp
            if stamp.sec == 0 and stamp.nanosec == 0:
                stamp_sec = stamp_ns * 1e-9
            else:
                stamp_sec = stamp.sec + stamp.nanosec * 1e-9
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            out.write(
                f"{stamp_sec:.9f} {p.x:.9f} {p.y:.9f} {p.z:.9f} "
                f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="input ROS 2 bag directory")
    parser.add_argument("-t", "--topic", default="/glim_ros/odom_scanend_corrected")
    parser.add_argument("-o", "--output", required=True, help="output TUM trajectory txt")
    args = parser.parse_args()

    count = export_odometry(args.bag, args.topic, args.output)
    print(f"exported {count} odometry messages from {args.topic} -> {args.output}")


if __name__ == "__main__":
    main()
