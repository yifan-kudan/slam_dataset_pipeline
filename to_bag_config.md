# Config-driven bag conversion

Generic entry point:

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 process/to_bag.py 20230531 -o 20230531_bag -c process/bag_conpr.yaml
```

Compatibility wrappers keep the old commands working:

```bash
/usr/bin/python3 process/conpr_to_bag.py 20230531 -o 20230531_bag
/usr/bin/python3 process/conpr_to_bag_complete.py 20230531 -o 20230531_complete_bag

source /home/kudan/prj/ws_livox/install/setup.bash
/usr/bin/python3 process/conpr_to_bag_fastlio.py 20230531 -o 20230531_fastlio_bag
```

## Configs

- `bag_conpr.yaml`: ConPR `.bin` -> `/livox/points` `sensor_msgs/PointCloud2`
- `bag_conpr_complete.yaml`: ConPR `.bin` -> PointCloud2 with `tag` and `line`
- `bag_conpr_fastlio.yaml`: ConPR `.bin` -> `/livox/lidar` `livox_ros_driver2/CustomMsg`
- `bag_pcd_xyzi_example.yaml`: template for datasets with `pcd/TIME_STAMP.pcd`, `imu.csv`, and `gt.csv`

## Important fields

`dataset.lidar` describes the source point cloud files:

```yaml
dir: LiDAR
pattern: "*.bin"
format: conpr_livox_bin   # or pcd_xyzi
timestamp_unit: ns
time_shift_sec: -0.000705
```

`bag.lidar` describes the output ROS topic and message:

```yaml
topic: /livox/points
message: pointcloud2      # or livox_custom
fields_preset: xyzi_t     # xyzi, xyzi_t, xyzi_t_tag_line
```

`bag.gt.orientation` controls how odometry orientation is filled:

- `nearest_imu`: use nearest IMU quaternion, useful for ConPR `pose.csv` with only `time,x,y,z`
- `row`: use columns 4:8 from GT, useful for `gt.csv` with `time,x,y,z,qx,qy,qz,qw`

For PCD XYZI datasets, start from `bag_pcd_xyzi_example.yaml` and usually change `frames`, topic names, timestamp units, and `tf_static.matrix`.
