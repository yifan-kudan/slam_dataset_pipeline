#!/bin/bash
# One kd_localizer run on a Hilti 2023 sequence. Runs INSIDE the release container:
#
#   scripts/hilti/in_container.sh $SDP/scripts/hilti/run_seq.sh site2_robot_1
#   scripts/hilti/in_container.sh $SDP/scripts/hilti/run_seq.sh site2_robot_1 1 120
#                                                              (rate) (first 120 s only)
#
# NAME picks which config to run and what the artefacts are called, defaulting to the
# sequence. A cross-sequence run replays the SAME bag against ANOTHER sequence's map, so
# it needs its own params/cfg and must not overwrite the own-map artefacts:
#
#   NAME=site1_handheld_1_in_site1_handheld_4 \
#     scripts/hilti/in_container.sh $SDP/scripts/hilti/run_seq.sh site1_handheld_1
#
# in_container.sh supplies SDP (this checkout), W (the results dir) and HILTI_ROOT (the
# bag tree); the defaults below are the mount points it uses, so the script also runs
# by hand in a container that already has them.
#
# Plays the LiDAR+IMU+tf_static bag on the local SSD (see config/hilti/
# rebag_hilti_odom_{handheld,robot}.yaml), not the ~30-65 GB camera bag: `ros2 bag play`
# reads every message out of storage regardless of who subscribes.
#
# -r 1 and not slower. The node has a 20 Hz process_frequency, so playing faster discards
# frames; playing slower buys nothing here because kd_localizer, unlike FAST-LIO, is not
# on SensorDataQoS and does not drop clouds under load.
set -u
SEQ=$1; RATE=${2:-1}; LIMIT=${3:-0}

NAME=${NAME:-$SEQ}
SDP=${SDP:-/opt/sdp}
W=${W:-/work}
BAG=${HILTI_ROOT:?set HILTI_ROOT to the bag tree (in_container.sh does)}/${SEQ}_fastlio_bag
# The same Fast DDS shared-memory profile batch_fastlio.py exports, for the same reason and
# now for a second one. A Hilti cloud is 1.37 MB, kd_localizer_node subscribes with
# rclcpp::SensorDataQoS() -- best effort, KeepLast(5) -- and the default 512 KB SHM segment
# cannot hold that stream, so the writer drops rather than waits. With ONE reader the
# localizer got away with it. Adding kd_relocalizer_node puts a SECOND best-effort reader on
# the same topic, one that stops draining its port for seconds at a time while a BBS search
# runs, and site2_robot_2 measured 1008 unique poses against 1562 without it -- 35 % of the
# run's frames gone, and a LOST at 20 s on a bag that had none. Exported here so the node,
# the relocalizer and `ros2 bag play` (the writer) all inherit it from one place.
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-$SDP/config/dds/fastdds_shm_big.xml}

# ...and reap the segments the last runs leaked, or the profile above silently stops
# working. The teardown at the end of this script SIGKILLs the nodes, so Fast DDS never
# unlinks its shared-memory files: every run leaves ~128 MB behind in /dev/shm, and after
# roughly fifty runs a 7.7 GB tmpfs is full. The failure mode is not an error the run
# reports -- the participant logs `Failed to create segment` once, drops SHM, falls back to
# UDP, and delivers 4.5 Hz of a 10 Hz cloud stream while looking like a normal run. It
# invalidated a DDS control set here before it was noticed.
#
# -mmin +30 rather than a blanket delete: runs are sequential and none lasts 30 minutes, so
# anything older cannot belong to a live run of ours -- and this is the HOST's /dev/shm
# (compose mounts /dev), so a blanket delete could pull the floor out from under an
# unrelated ROS node on the host.
find /dev/shm -maxdepth 1 -name 'fastrtps_*' -mmin +30 -delete 2>/dev/null
echo "$NAME: /dev/shm $(df -h /dev/shm | awk 'NR==2{print $3"/"$2" used"}')"

# Bags from this pipeline are already rosbag2 metadata v8, which Humble reads, so
# no downgrade step -- unlike bags recorded natively on this Jazzy host.
DUR=$(python3 -c "import yaml;print(int(yaml.safe_load(open('$BAG/metadata.yaml'))['rosbag2_bagfile_information']['duration']['nanoseconds']/1e9/$RATE)+60)")
[ "$LIMIT" != "0" ] && DUR=$((LIMIT + 60))

# 10 of the 15 bags were rebagged with config/hilti/rebag_hilti_fastlio.yaml, which DROPS
# /tf_static (FAST-LIO reads its extrinsic from its own config and ignores tf).
# kd_localizer cannot: with base_frame_id imu_link and clouds stamped `velodyne` it must
# look the extrinsic up in TF, and without it the node sits at UNINITIALIZED forever.
# Publish it here rather than rewriting ~30 GB of bags. The values are per platform in
# config/hilti/platforms.yaml, verified against both the /tf_static the newer bags carry
# and the extrinsic_T in config/hilti/fast_lio_hilti*.yaml.
if ! grep -q "/tf_static" "$BAG/metadata.yaml"; then
  echo "$NAME: bag has no /tf_static, publishing imu_link -> velodyne"
  # shellcheck disable=SC2046
  ros2 run tf2_ros static_transform_publisher \
    $(python3 "$SDP/src/hilti/dataset.py" "$SEQ" --tf-args) > "$W/tf_${NAME}.log" 2>&1 &
fi

# A relocalizer run (src/hilti/gen_reloc.py) moves base_frame_id off imu_link onto a
# gravity-levelled `blevel`, because the 3D-BBS roll/pitch search is +-max_tilt about
# LEVEL and imu_link is 90-180 deg from it in these maps. Pure rotation, same origin.
if [ -f "$W/tf_blevel_${NAME}.args" ]; then
  echo "$NAME: publishing imu_link -> blevel (levelled base frame)"
  # shellcheck disable=SC2046
  ros2 run tf2_ros static_transform_publisher $(cat "$W/tf_blevel_${NAME}.args") \
    > "$W/tfb_${NAME}.log" 2>&1 &
fi

python3 "$SDP/src/hilti/monitor_pose.py" -o "$W/run_${NAME}.csv" --seconds "$DUR" \
  > "$W/mon_${NAME}.log" 2>&1 &

ros2 launch kd_amr_ros localizer.launch.py params_file:="$W/params_${NAME}.yaml" \
  lidar_topic:=/velodyne_points imu_topic:=/imu use_relocalizer:=false use_sim_time:=true \
  > "$W/node_${NAME}.log" 2>&1 &
LAUNCH=$!
sleep 12

# RELOC=1 starts the shipped 3D-BBS relocalizer. Not localizer.launch.py's
# use_relocalizer: that starts it with the localizer, and it answers the startup
# UNINITIALIZED window with a single-frame global search whose hint would replace a
# perfectly good initial_pose. RELOC_DELAY (seconds of wall clock after bag play begins)
# is how an own-map recovery test keeps the prior and still gets BBS on a mid-run LOST;
# a cold-start run sets it to 0 because UNINITIALIZED is exactly what it is asking BBS.
if [ "${RELOC:-0}" = "1" ]; then
  (
    sleep "${RELOC_DELAY:-0}"
    echo "starting kd_relocalizer_node after ${RELOC_DELAY:-0}s"
    exec ros2 run kd_amr_ros kd_relocalizer --ros-args \
      -r __node:=kd_relocalizer_node \
      --params-file "$W/params_${NAME}.yaml" \
      -r lidar:=/velodyne_points \
      -p use_sim_time:=true
  ) > "$W/reloc_${NAME}.log" 2>&1 &
fi

if [ "$LIMIT" != "0" ]; then
  timeout "$LIMIT" ros2 bag play "$BAG" --clock 100 -r "$RATE" > "$W/play_${NAME}.log" 2>&1
else
  ros2 bag play "$BAG" --clock 100 -r "$RATE" > "$W/play_${NAME}.log" 2>&1
fi
sleep 4
kill -INT $LAUNCH 2>/dev/null
sleep 3
pkill -f monitor_pose 2>/dev/null
pkill -f static_transform_publisher 2>/dev/null
pkill -9 -f kd_relocalizer 2>/dev/null
pkill -9 -f kd_localizer 2>/dev/null
sleep 2

python3 - "$W/run_${NAME}.csv" <<'PY'
import collections, csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
c = collections.Counter(r["status"] for r in rows)
tr = [r for r in rows if r["status"] == "TRACKING" and r["pose_stamp_s"]]
print(f"[{sys.argv[1]}] {dict(c)}", end=" ")
if tr:
    print(f"tracked {float(tr[-1]['pose_stamp_s']) - float(tr[0]['pose_stamp_s']):.1f} s bag time",
          end=" ")
lost = [r for r in rows if r["status"] == "LOST"]
print(f"first lost: {lost[0]['lost_reason']} @ {lost[0]['stamp_s']} "
      f"inlier {lost[0]['ray_inlier_ratio']}" if lost else "")
PY

# Did this run actually get shared memory? A participant that fails to create its segment
# logs once, drops SHM, falls back to UDP and then looks like a normal run at half the
# frame rate -- so the run has to say so itself rather than be audited later.
if grep -q 'Failed to create segment' "$W/node_${NAME}.log" 2>/dev/null; then
  echo "[INVALID] $NAME: Fast DDS could not create its SHM segment -- this run fell back" \
       "to UDP and its frame rate is not comparable. /dev/shm was probably full."
fi

# The relocalizer's own verdict. Every BBS attempt logs either a score and a time or the
# reason it gave up, and that is the measurement a relocalizer run exists to produce --
# the monitor CSV only sees a hint after it has already been applied.
if [ -f "$W/reloc_${NAME}.log" ]; then
  L=$W/reloc_${NAME}.log
  echo "[reloc] $(grep -c 'starting BBS' "$L") attempts," \
       "$(grep -c 'succeeded' "$L") succeeded," \
       "$(grep -c 'timed out' "$L") timed out," \
       "$(grep -c 'below threshold' "$L") below threshold"
  grep -E 'succeeded|Published 3D pose hint|Failed to load|Loaded BBS map|what\(\)' "$L" \
    | head -8
fi
