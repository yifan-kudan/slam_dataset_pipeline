# slam_dataset_pipeline
This repository is created for Research on datasets for lifelong localization challenge in unstructured dynamic construction environments

## File structure

```
SLAM_DATASET_PIPELINE
|   README.md
└───config
|   └───dataset_configuration_folder
|       |   bag_convertion_config.yaml
|       |   SLAM_system_config.yaml
|       |   ...
|
└───src
|   |   to_bag.py
|   |   rosbag1_to_ros2.py
|   |   read_tf_extrinsic.py
|   |       
```

## Inspect ROS bag

Introspect a ROS 1 or ROS 2 bag: topics, message types, rates, timestamp layout
and (for PointCloud2 / Imu) the per-message structure a converter config needs.

The goal is onboarding: point this at a new dataset's raw bag and it tells you
what you must declare in a ``bag_*.yaml`` config -- which topic is LiDAR / IMU,
the PointCloud2 field layout (does it carry ``time`` / ``ring``?), whether stamps
are absolute Unix time or sensor-relative, and where the timestamps cluster or
gap. It reads with the pure-python ``rosbags`` library (no ROS environment), and
``AnyReader`` auto-detects ROS 1 vs ROS 2.

Two passes:
  * cheap pass over every message using the bag receive-time (no deserialize) for
    per-topic count / rate / dt distribution / gaps;
  * a sample pass that deserializes the first ``--sample`` messages per topic to
    read structure (PointCloud2 fields, header presence, header-vs-bag-time
    offset, per-point time range, frame_id).
With ``--deep`` the sample pass deserializes *every* message so the header-stamp
dt / monotonicity / gap stats cover the whole bag (slow on large bags).

Examples:
    python3 src/inspect_bag.py "/path/ConSLAM/seq1/Copy of recording.bag"
    python3 src/inspect_bag.py /path/seq1_fastlio_bag --topics /velodyne_points /imu
    python3 src/inspect_bag.py /path/raw.bag --deep --json > seq1_probe.json

## ROS bag convertion

some
datasets ship their raw sensor stream as a single ROS 1 bag. ConSLAM is one:
its ``recording.bag`` holds LiDAR (``sensor_msgs/PointCloud2``), IMU
(``sensor_msgs/Imu``) and camera images. FAST-LIO only needs LiDAR + IMU.

This tool reads the ROS 1 bag with the pure-python ``rosbags`` library (no ROS
environment needs to be sourced), keeps only the topics named in the config,
optionally renames topics / frame ids, and for Velodyne-style clouds rebases the
per-point ``time`` field so it starts at 0 (with the header stamp adjusted so
absolute point times are preserved). The result is a ROS 2 bag that plays back
just like the ConPR ``*_fastlio_bag`` bags.

Example:
    python3 src/rosbag1_to_ros2.py \
        "/path/ConSLAM/seq1/Copy of recording.bag" \
        -o /path/ConSLAM/seq1_fastlio_bag \
        -c config/conslam/bag_conslam_fastlio.yaml