#!/bin/bash
# One KdLocalizer run, INSIDE the localizer container. Normally started by campaign.sh:
#
#   run_one.sh <run dir>              runs/<name>/ as written by src/kdloc/make_runs.py
#   LIMIT=60 run_one.sh <run dir>     play only the first 60 s of the bag (smoke test)
#
# Reads <run dir>/run.env and writes into <run dir>/out/:
#   pose.csv       /localization_status + pose, one row per status message (src/kdloc/monitor.py)
#   node.log       localizer;  reloc.log  relocalizer;  play.log  ros2 bag play
#   traj.tum       the localizer's saved trajectory;  save.log  the service call
#   tf_<k>.log     one per static transform;  monitor.log
#   bag_metadata.yaml, limit_s, play_start_wall, play_end_wall
#
# Steps: publish the static transforms, start the pose monitor, start the localizer and wait
# until it is active, start the relocalizer, play the bag in real time, save the trajectory,
# stop everything and print a summary (src/kdloc/logs.py summary). A relocalizer with
# RELOC_DELAY 0 (cold start) starts before the bag and is waited on until its BBS map is
# loaded, so a slow map load cannot cost the first frames. Otherwise it starts RELOC_DELAY s
# after the bag, so it cannot answer the start-up UNINITIALIZED window and replace a good
# initial pose. Exit status 0 when the run produced a pose log.
#
# The campaign directory and this checkout must be mounted at the same paths inside the
# container as on the host, because run.env and params.yaml carry absolute paths.
set -u
RUN_DIR=$(cd "${1:?usage: run_one.sh <run dir>}" && pwd)
OUT=$RUN_DIR/out
SDP=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LIMIT_OVERRIDE=${LIMIT:-}
# shellcheck disable=SC1091
source "$RUN_DIR/run.env"
[ -n "$LIMIT_OVERRIDE" ] && LIMIT=$LIMIT_OVERRIDE
mkdir -p "$OUT"
echo "$LIMIT" > "$OUT/limit_s"

set +u
source "${ROS_SETUP:-/opt/ros/humble/setup.bash}"
source "${KD_SETUP:-$HOME/ros2_ws/install/setup.bash}"
set -u
export ROS_DOMAIN_ID
# A large-cloud shared-memory profile: with the default 512 KB segment the Fast DDS writer
# drops clouds rather than waits, silently (see config/dds/).
export FASTRTPS_DEFAULT_PROFILES_FILE=$DDS_PROFILE
# Every run ends by SIGKILLing its nodes, so Fast DDS never unlinks its segments; reap the
# ones older than any run (ipc: host makes this the host's /dev/shm, so not a blanket delete).
find /dev/shm -maxdepth 1 -name 'fastrtps_*' -mmin +30 -delete 2>/dev/null
echo "$NAME: $SEQUENCE in map $MAP_NAME, relocalizer +${RELOC_DELAY} s," \
     "/dev/shm $(df -h /dev/shm | awk 'NR==2{print $3"/"$2}')"

[ -f "$MAP" ] || { echo "[FAILED] $NAME: no map $MAP"; exit 1; }
[ -f "$BAG/metadata.yaml" ] || { echo "[FAILED] $NAME: no bag $BAG"; exit 1; }
cp "$BAG/metadata.yaml" "$OUT/bag_metadata.yaml"
DUR=$(python3 -c "import yaml;print(int(yaml.safe_load(open('$BAG/metadata.yaml'))['rosbag2_bagfile_information']['duration']['nanoseconds']/1e9/$PLAY_RATE)+120)")
[ "$LIMIT" != "0" ] && DUR=$((LIMIT + 120))

PIDS=()
k=0
if [ -f "$RUN_DIR/static_tf.args" ]; then
  while read -r tf_args; do
    [ -n "$tf_args" ] || continue
    k=$((k + 1))
    # shellcheck disable=SC2086
    ros2 run tf2_ros static_transform_publisher $tf_args > "$OUT/tf_$k.log" 2>&1 &
    PIDS+=($!)
  done < "$RUN_DIR/static_tf.args"
elif ! grep -q "/tf_static" "$BAG/metadata.yaml"; then
  echo "[WARN] $NAME: the bag has no /tf_static and the campaign publishes no static transform"
fi

python3 "$SDP/src/kdloc/monitor.py" -o "$OUT/pose.csv" --seconds "$DUR" > "$OUT/monitor.log" 2>&1 &
PIDS+=($!)

stop_all() {
  kill "${PIDS[@]}" 2>/dev/null
  pkill -f static_transform_publisher 2>/dev/null
  pkill -9 -f kd_relocalizer 2>/dev/null
  pkill -9 -f kd_localizer 2>/dev/null
}

ros2 launch kd_amr_ros localizer.launch.py params_file:="$PARAMS" \
  lidar_topic:="$LIDAR_TOPIC" imu_topic:="$IMU_TOPIC" use_relocalizer:=false use_sim_time:=true \
  > "$OUT/node.log" 2>&1 &
LAUNCH=$!
t0=$(date +%s)
until ros2 lifecycle get /kd_localizer_node 2>/dev/null | grep -q "^active"; do
  if [ $(( $(date +%s) - t0 )) -gt "$STARTUP_TIMEOUT" ] || ! kill -0 $LAUNCH 2>/dev/null; then
    echo "[FAILED] $NAME: localizer not active after $(( $(date +%s) - t0 )) s"
    tail -5 "$OUT/node.log"
    stop_all
    exit 1
  fi
  sleep 1
done
echo "$NAME: localizer active after $(( $(date +%s) - t0 )) s"

start_relocalizer() {
  exec ros2 run kd_amr_ros kd_relocalizer --ros-args -r __node:=kd_relocalizer_node \
    --params-file "$PARAMS" -r lidar:="$LIDAR_TOPIC" -p use_sim_time:=true
}
if [ "$RELOC_DELAY" = "0" ]; then
  (start_relocalizer) > "$OUT/reloc.log" 2>&1 &
  t0=$(date +%s)
  until grep -qE "Loaded BBS map|Failed to load|what\(\)" "$OUT/reloc.log" 2>/dev/null; do
    if [ $(( $(date +%s) - t0 )) -gt "$STARTUP_TIMEOUT" ]; then
      echo "[WARN] $NAME: BBS map not loaded after $STARTUP_TIMEOUT s"
      break
    fi
    sleep 1
  done
  echo "$NAME: relocalizer ready after $(( $(date +%s) - t0 )) s:" \
       "$(grep -m1 -E 'Loaded BBS map|Failed to load|what\(\)' "$OUT/reloc.log" | sed 's/.*\] //')"
else
  ( sleep "$RELOC_DELAY"; echo "starting kd_relocalizer_node after ${RELOC_DELAY} s"; start_relocalizer ) \
    > "$OUT/reloc.log" 2>&1 &
fi

date +%s.%N > "$OUT/play_start_wall"
if [ "$LIMIT" != "0" ]; then
  timeout "$LIMIT" ros2 bag play "$BAG" --clock 100 -r "$PLAY_RATE" > "$OUT/play.log" 2>&1
else
  ros2 bag play "$BAG" --clock 100 -r "$PLAY_RATE" > "$OUT/play.log" 2>&1
fi
date +%s.%N > "$OUT/play_end_wall"
sleep 4
timeout 20 ros2 service call /kd_localizer_node/save_trajectory kd_amr_ros_interfaces/srv/SaveFile \
  "{file_path: '$OUT/traj.tum'}" > "$OUT/save.log" 2>&1
kill -INT $LAUNCH 2>/dev/null
sleep 3
stop_all
sleep 2

python3 "$SDP/src/kdloc/logs.py" summary "$RUN_DIR"
[ -s "$OUT/pose.csv" ] || exit 3
