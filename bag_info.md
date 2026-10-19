# Go2 PRO bags — 2026-10-08

Recorded on the Jetson AGX Orin from a Unitree Go2 PRO over the office Wi-Fi (WebRTC), with the upstream ROS 2
driver [go2_ros2_sdk](https://github.com/abizovnuralem/go2_ros2_sdk) (commit `b440609`, ROS 2 Humble, in Docker)
plus local changes: the camera is kept as H264, four topics the driver normally drops are published, and the lidar
can be stored exactly as the robot sends it.

- Bags: laptop `~/prj/go2pro/demo_8_10/bags/` (also on the Jetson in `~/prj/go2_demo/bags/`)
- Setup, local changes and how to record: `~/prj/go2pro/demo_8_10/PROGRESS.md`

## Recommended bag

**`go2_20261008_154551`**: 58.7 s, robot walking (3.2 m path, up to 0.55 m/s), MCAP, 16 MB. No gaps in any topic.
The lidar is stored as the robot sends it (compressed voxels).
**`go2_20261008_154551_cloud`** (67 MB) is the same bag with `/point_cloud2` added: one point at the centre of each
occupied voxel, made offline with `voxel_to_cloud.sh`.

## Topics in the recommended bag

Rate = number of messages ÷ time from the first to the last message. Longest interval = the largest time between two
consecutive messages, using the time each message was recorded.

| Topic | Message type | Messages | Rate | Longest interval | Content |
|---|---|---|---|---|---|
| `/utlidar/voxel_map_compressed` | `go2_interfaces/msg/VoxelMapCompressed` | 420 | 7.3 Hz | 0.17 s | Lidar as the robot sends it: compressed voxel map, 5 cm voxels, 128 × 128 × 38 voxels around the robot |
| `/point_cloud2` (only in `_cloud`) | `sensor_msgs/msg/PointCloud2` | 420 | 7.3 Hz | 0.17 s | One point at the centre of each occupied voxel: x, y, z (float32), frame `odom`, median 11,600 points per message (= the number of occupied voxels) |
| `/imu` | `go2_interfaces/msg/IMU` | 1146 | 20.0 Hz | 0.08 s | Body IMU: orientation, angular rate, acceleration, roll/pitch/yaw, temperature. **No header** |
| `/go2_states` | `go2_interfaces/msg/Go2State` | 1147 | 20.0 Hz | 0.08 s | Robot state: position, velocity, body height. The other fields (mode, gait, foot force, foot positions and speeds, obstacle ranges) are always 0, also while walking |
| `/odom` | `nav_msgs/msg/Odometry` | 1073 | 18.7 Hz | 0.08 s | Pose of `base_link` in `odom`, from the robot |
| `/tf` | `tf2_msgs/msg/TFMessage` | 1141 | 19.4 Hz | 1.0 s | `odom` → `base_link` with every `/odom` message, plus the 12 leg joints once a second. The 1.0 s interval is only at the start: the first `odom` transform comes 1.4 s after recording starts |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 1 | – | – | Fixed transforms of the robot model |
| `/robot_description` | `std_msgs/msg/String` | 1 | – | – | Robot model (URDF) |
| `/joint_states` | `sensor_msgs/msg/JointState` | 57 | 1.0 Hz | 1.0 s | 12 joint angles (no velocity, no torque) |
| `/lowstate` | `go2_interfaces/msg/LowState` | 57 | 1.0 Hz | 1.0 s | Motor angles and temperatures, battery, supply voltage (only some fields are filled, see below). Its `foot_force` is not a measurement: the Go2 PRO has no foot force sensors |
| `/battery` | `sensor_msgs/msg/BatteryState` | 57 | 1.0 Hz | 1.0 s | Voltage, current, charge, temperature (taken from the same robot message as `/lowstate`) |
| `/utlidar/lidar_state` | `go2_interfaces/msg/LidarState` | 286 | 5.0 Hz | 0.21 s | Lidar status: its point cloud frame rate (~14.7 Hz), packet loss, its IMU rate (~250 Hz), firmware versions |
| `/multiplestate` | `std_msgs/msg/String` | 57 | 1.0 Hz | 1.0 s | Robot settings as JSON: brightness, obstacle avoidance on/off (on), UWB on/off, volume |
| `/camera/h264` | `sensor_msgs/msg/CompressedImage` | 819 | 14.3 Hz | 0.11 s | Front camera, 1280 × 720, the H264 stream as the robot sends it (`format` = `"h264"`); a full key frame about every 2.5 s |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 819 | 14.3 Hz | 0.11 s | Camera calibration (the driver's 720p calibration file) |

The other bags differ as follows: before 15:45 the lidar was decoded on the Jetson while recording, so those bags
have `/point_cloud2` and `/scan` (`sensor_msgs/msg/LaserScan`, a 2D scan cut from the point cloud) instead of
`/utlidar/voxel_map_compressed`. Their `/point_cloud2` is built differently from the one in `_cloud` bags: the
driver takes the corners of the small cubes the phone app draws for the occupied voxels, so each point is a cube
corner, not a voxel centre, and the same corner often appears more than once. Its `intensity` field is the app's
colour by height, not the strength of the lidar return. The driver also drops points whose colour value is 0, which
loses about 5 % of the occupied voxels. `go2_20261008_133750` has the decoded camera image `/camera/image_raw`
(`sensor_msgs/msg/Image`, bgr8) instead of `/camera/h264`. `/lowstate`, `/battery`, `/utlidar/lidar_state` and
`/multiplestate` exist only from 15:24 on.

## Go2 message types (`go2_interfaces`)

These types are Go2 specific. The MCAP bags (15:45 and later) contain their definitions, so `rosbags` reads them with
no setup. For the older sqlite3 bags, register the `.msg` files from `go2_ros2_sdk/go2_interfaces/msg/`.
`ros2 bag play` needs the `go2_interfaces` package built for these topics.

**`IMU`**: `float32[4] quaternion`, `float32[3] gyroscope` (rad/s), `float32[3] accelerometer` (m/s², ~9.5 on z at
rest), `float32[3] rpy` (rad), `int8 temperature`.

**`Go2State`**: `uint8 mode`, `int32 progress`, `uint8 gait_type`, `float32 foot_raise_height`,
`float32[3] position`, `float32 body_height` (m, ~0.32 standing, ~0.07 lying), `float32[3] velocity`,
`float32[4] range_obstacle`, `int16[4] foot_force`, `float32[12] foot_position_body`,
`float32[12] foot_speed_body`. The robot fills only `position`, `body_height` and `velocity`; every other field is
always 0, also while it walks, so this message does not show the current gait.

**`LowState`**: the type has many fields, but the robot fills only these over Wi-Fi (the rest stay 0):
- `foot_force` `int16[4]`, order FR, FL, RR, RL: filled with numbers, but the Go2 PRO has no foot force sensors
  (Unitree fits them only to the Go2 EDU; the message type is the same for all Go2 models), so it is not a measured
  foot load. Unitree does not document what the PRO puts here. The four values move slowly and together, not with
  the steps: 104–110 the whole time while walking in the recommended bag; in `go2_20261008_152446` (robot standing
  the whole minute) all four rise together from about 90 to about 102 while the board temperature goes from 36 to
  37 °C. Most likely it is the reading of a sensor input with nothing connected; a test of several minutes lying
  down and then standing would confirm it.
- `motor_state[0..11]`: `q` (angle, rad), `temperature` (°C), `lost`; motors 0–2 = FR hip/thigh/calf,
  3–5 = FL, 6–8 = RR, 9–11 = RL
- `bms_state`: `soc` (charge, %), `current` (mA, negative = discharging), `cycle`, `bq_ntc` and `mcu_ntc` (°C)
- `power_v` (V), `temperature_ntc1` (°C), `imu_state.rpy`

**`LidarState`**: `float64 stamp`, `firmware_version`, `software_version`, `sdk_version` (strings),
`sys_rotation_speed`, `com_rotation_speed`, `uint8 error_state`, `cloud_frequency` (Hz), `cloud_packet_loss_rate` (%),
`cloud_size`, `cloud_scan_num`, `imu_frequency` (Hz), `imu_packet_loss_rate`, `float32[3] imu_rpy`,
`serial_recv_stamp`, `serial_buffer_size`, `serial_buffer_read`.

**`VoxelMapCompressed`**: `float64 stamp`, `string frame_id`, `float64 resolution` (0.05 m), `float64[3] origin`,
`int16[3] width` (128, 128, 38), `uint64 src_size`, `uint8[] data` (compressed voxels).
`data` can only be decompressed with `libvoxel.wasm` from the driver (Unitree does not publish the compression
format). Decompressed, it is `src_size` = 77,824 bytes = 128 × 128 × 38 bits, one bit per voxel: 1 means at least one
lidar return fell inside it. Bits are read most significant bit first, x varies fastest and z slowest. Voxel
(i, j, k) covers `origin + [i, i + 1] × resolution` along x, and the same for y and z. Only occupancy is stored: the
measured points, how many there were, their return strength and their measurement time are lost.

## Timestamps

- `header.stamp` of every message is the Jetson clock when the driver published the message (NTP-synced), not
  the time the robot measured it.
- `/imu`, `/lowstate` and `/utlidar/lidar_state` have no header: use the time the message was recorded.
- `/point_cloud2` in `_cloud` bags: the time the voxel message it was decoded from was recorded.
- `/camera/h264`: the camera's own frame clock, started at the Jetson clock on the first frame. It runs about 5 %
  fast (the stamps step as if the camera ran at 15 fps; it delivers 14.3 fps), about 3 s over a minute. Use the
  recording time for the real time.
- The `stamp` field inside `VoxelMapCompressed` and `LidarState` is the robot's, but rounded to about 128 s: not usable.
- The robot's own measurement times (sent with the IMU/state and pose) are not stored yet.

## What the Go2 does not give over Wi-Fi

- **Raw lidar scans and the lidar's IMU.** The lidar itself outputs point cloud frames at 14.7 Hz (21,600 points per
  second, the rated sample rate of the Unitree L1) with a 250 Hz IMU, but the robot only sends the 7.3 Hz voxel map
  (half the lidar frame rate) and the 20 Hz body IMU. Subscribing to other topic names
  (`rt/utlidar/cloud`, `rt/utlidar/cloud_deskewed`, `rt/utlidar/imu`, `rt/lowstate`, `rt/odommodestate`,
  `rt/utlidar/robot_odom`, `rt/utlidar/height_map_array`) returned no messages.
- **Foot force.** The Go2 PRO has no foot force sensors (see `LowState` above), and without joint torques the load on
  the feet cannot be estimated either.
- **Joint velocities and torques.**
- **Lidar while lying down.** With the robot lying down it sends no lidar data at all.

## All bags

| Bag | Length | Size | Format | Robot (path from `/odom`) | Lidar in the bag | Camera | Notes |
|---|---|---|---|---|---|---|---|
| `go2_20261008_133452` | 30.5 s | 118 MB | sqlite3 | standing still (0.02 m) | `/point_cloud2` 5.7 Hz | none | Jetson at 50 W |
| `go2_20261008_133750` | 30.1 s | 1137 MB | sqlite3 | standing still (0.02 m) | `/point_cloud2` 1.5 Hz | `/camera/image_raw` 13.2 Hz | Decoding the camera slowed the lidar decoding |
| `go2_20261008_134246` | 30.2 s | 61 MB | sqlite3 | standing still (0.02 m) | `/point_cloud2` 2.5 Hz, one 6 s gap | `/camera/h264` 13.9 Hz | Camera stamps wrong (wrong clock assumed); do not use |
| `go2_20261008_134519` | 61.2 s | 75 MB | sqlite3 | standing still (0.03 m) | `/point_cloud2` 1.4 Hz | `/camera/h264` 14.3 Hz | Low lidar rate not explained |
| `go2_20261008_134716` | 60.9 s | 86 MB | sqlite3 | standing still (0.03 m) | `/point_cloud2` 1.9 Hz | none | Low lidar rate not explained |
| `go2_20261008_135332` | 59.5 s | 13 MB | sqlite3 | **lying down** | none (robot sends none) | `/camera/h264` 14.3 Hz | IMU evenly at 20 Hz |
| `go2_20261008_135750` | 59.8 s | 119 MB | sqlite3 | standing still (0.06 m) | `/point_cloud2` 6.95 Hz, no gaps | `/camera/h264` 14.3 Hz | Good bag of a still robot |
| `go2_20261008_150827` | 58.1 s | 151 MB | sqlite3 | standing (0.48 m) | `/point_cloud2` 7.3 Hz, no gaps | `/camera/h264` 14.3 Hz | Good bag of a still robot |
| `go2_20261008_152446` | 59.1 s | 123 MB | sqlite3 | **walking (3.3 m)** | `/point_cloud2` 2.9 Hz, 36 gaps | `/camera/h264` 14.3 Hz | First with `/lowstate` etc.; decoding on the Jetson could not keep up while walking |
| **`go2_20261008_154551`** | 58.7 s | 16 MB | **MCAP** | **walking (3.2 m)** | `/utlidar/voxel_map_compressed` 7.3 Hz, no gaps | `/camera/h264` 14.3 Hz | **Recommended.** All topics, no gaps |
| **`go2_20261008_154551_cloud`** | 58.7 s | 67 MB | **MCAP** | same | the above + `/point_cloud2` 7.3 Hz | same | Offline-decoded version of the above, one point per occupied voxel |

In the decoded-lidar bags, `/imu` and `/odom` arrive in bursts whenever the lidar decoding is slow (same average rate,
uneven spacing); this does not happen in `go2_20261008_135332` (no lidar) or `go2_20261008_154551` (no decoding).
`rtc_probe/summary.json` in the same folder lists every topic the robot sends over Wi-Fi, with message counts and a
sample of each.
