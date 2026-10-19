#!/usr/bin/env python3
"""Count what actually reaches a subscriber, to separate transport loss from a slow node.

    count_topics.py --seconds 70

Subscribes to /velodyne_points with reliable + keep_all (so this counter itself cannot
lose anything) and to /Odometry, which FAST-LIO publishes once per processed scan. If
clouds arrive at the play rate while /Odometry lags, the node is the bottleneck; if both
lag, the loss is upstream of the node.
"""
import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilti import dataset                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=70)
a = ap.parse_args()

rclpy.init()
n = Node("count_topics")
n.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
q = QoSProfile(history=HistoryPolicy.KEEP_ALL, reliability=ReliabilityPolicy.RELIABLE, depth=1)
c = {"cloud": 0, "odom": 0, "imu": 0, "first": None, "last": None}


def on_cloud(m):
    c["cloud"] += 1
    t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    c["first"] = c["first"] or t
    c["last"] = t


n.create_subscription(PointCloud2, dataset.CLOUD_TOPIC, on_cloud, q)
n.create_subscription(Odometry, "/Odometry", lambda m: c.update(odom=c["odom"] + 1), 50)
n.create_subscription(Imu, dataset.IMU_TOPIC, lambda m: c.update(imu=c["imu"] + 1), q)
t0 = time.time()
while time.time() - t0 < a.seconds:
    rclpy.spin_once(n, timeout_sec=0.1)
span = (c["last"] - c["first"]) if c["first"] else 0
print(f"clouds {c['cloud']} over {span:.1f} s of bag time ({time.time()-t0:.0f} s wall), "
      f"imu {c['imu']} ({c['imu']/max(span,1e-9):.0f} Hz bag time), odometry {c['odom']}  "
f"-> node processed {100*c['odom']/max(c['cloud'],1):.1f}% of what arrived")
