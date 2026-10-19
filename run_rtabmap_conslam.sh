#!/bin/bash
# Run RTAB-Map over one ConSLAM *_cam_bag and collect what happened.
#
#   bash run_rtabmap_conslam.sh <bag_dir> [play_rate] [playback_duration_sec]
#
# Starts the launch graph, plays the bag with --clock, records icp_odometry's output
# for later evaluation, then tears the graph down and prints what RTAB-Map reported.
#
# No `set -u`: /opt/ros/jazzy/setup.bash reads AMENT_TRACE_SETUP_FILES unguarded and
# would abort the script on the very first line.
BAG=${1:?usage: run_rtabmap_conslam.sh <bag_dir> [rate] [duration_sec]}
RATE=${2:-0.3}
DURATION=${3:-90}

REPO=/home/kudan/prj/slam_dataset_pipeline
LAUNCH=$REPO/config/conslam/rtabmap_conslam.launch.py
NAME=$(basename "$BAG")
OUT=${OUT_DIR:-/media/kudan/seagate-2tb/datasets/ConSLAM_bag_with_cam/rtabmap/$NAME}
DB=$OUT/rtabmap.db
LOG=$OUT/rtabmap.log

# Anything left from an earlier run publishes on the same topics and silently
# corrupts this one -- two icp_odometry instances fighting over /icp_odom looks
# exactly like a working run in the log. Kill by full executable path so nothing
# unrelated is touched.
NODE_PATTERNS=(
  "rtabmap_conslam.launch.py"
  "rtabmap_odom/icp_odometry"
  "rtabmap_slam/rtabmap"
  "rtabmap_util/lidar_deskewing"
  "rtabmap_util/imu_to_tf"
  "rtabmap_sync/rgb_sync"
  "rtabmap_viz/rtabmap_viz"
)
kill_leftovers() {
  for p in "${NODE_PATTERNS[@]}"; do pkill -f "$p" 2>/dev/null; done
  # `ros2 bag record` is not part of the launch's process group, so it survives a
  # killed run, keeps holding the output directory, and its `pkill` in the next
  # run's teardown then kills THAT run's nodes. Match the recorder by its output
  # path so only this script's recorders are hit.
  pkill -f "bag record -o $OUT/odom_bag" 2>/dev/null
}

# Refuse to start if another instance is mid-run: its teardown would kill this run's
# graph half way through, and the symptom ("rtabmap received no data", 0 frames) looks
# nothing like a collision. A lock file rather than pgrep, which also matches this
# script's own process and the shell that launched it.
exec 9>/tmp/run_rtabmap_conslam.lock
if ! flock -n 9; then
  echo "!!! another run_rtabmap_conslam.sh holds the lock -- refusing to start"
  exit 1
fi

cleanup() {
  [ -n "$REC_PID" ] && kill -INT "$REC_PID" 2>/dev/null
  sleep 2
  [ -n "$LAUNCH_PID" ] && kill -INT -"$LAUNCH_PID" 2>/dev/null   # whole process group
  sleep 4
  [ -n "$LAUNCH_PID" ] && kill -TERM -"$LAUNCH_PID" 2>/dev/null
  kill_leftovers
  sleep 1
}
trap cleanup EXIT INT TERM

stale=$(pgrep -fc "rtabmap_odom/icp_odometry|rtabmap_slam/rtabmap" 2>/dev/null)
if [ "${stale:-0}" -gt 0 ]; then
  echo "!!! $stale stale rtabmap node(s) still running from an earlier run -- killing them first"
  kill_leftovers
  sleep 3
fi

# A fresh output directory every run: `ros2 bag record` refuses an existing one, and
# a half-overwritten database from a killed run would be read back as if it were this
# run's result.
rm -rf "$OUT"
mkdir -p "$OUT"

source /opt/ros/jazzy/setup.bash

echo "### $NAME  rate=$RATE duration=${DURATION}s  db=$DB" | tee "$LOG"

# setsid so the whole graph lands in its own process group and cleanup() can take it
# down in one signal instead of hunting children.
setsid ros2 launch "$LAUNCH" use_sim_time:=true viz:=false database_path:="$DB" \
  >>"$LOG" 2>&1 &
LAUNCH_PID=$!
setsid ros2 bag record -o "$OUT/odom_bag" --topics /icp_odom /rtabmap/mapPath \
  >>"$LOG" 2>&1 &
REC_PID=$!

# rtabmap subscribes lazily; a bag that starts before the graph is up loses its first
# scans (and icp_odometry then has no IMU history for them).
sleep 12

# READ-AHEAD MUST STAY SMALL on these bags. The queue counts MESSAGES, and each RGB
# frame is 9.56 MB, so the value the odometry bags use (batch_fastlio.py defaults to
# 10000) would try to buffer ~95 GB on a 15 GB machine. Measured with 2000: the player
# sat in uninterruptible I/O wait (state D) for 13 minutes having read 5 MB, published
# nothing, and took the whole graph down with it. 30 messages is ~285 MB.
ros2 bag play "$BAG" --clock -r "$RATE" --playback-duration "$DURATION" \
  --read-ahead-queue-size 30 2>&1 | tee -a "$LOG"

# Let the last scans finish before the graph goes away.
sleep 8
cleanup
trap - EXIT INT TERM

echo
echo "=== summary ==="
printf 'icp_odometry frames : %s\n' "$(grep -ac 'Odom: ratio' "$LOG")"
printf 'odom lost           : %s\n' "$(grep -ac 'Odom: ratio=0.000000' "$LOG")"
grep -aoE "Loop closure detected|Proximity detection|Rejected loop closure" "$LOG" | sort | uniq -c
grep -a "Did not receive data since" "$LOG" | tail -1
if [ -s "$DB" ]; then
  echo "=== rtabmap-report ==="
  rtabmap-report "$DB" 2>&1 | tail -8
  # Straight from the database, because the log lines are easy to mis-grep (this
  # used to report "0 updates" for a map that had 84 nodes in it). Link types:
  # 0 neighbour, 1 global loop closure, 2 local space, 9 gravity (from the IMU).
  echo "=== map contents ==="
  python3 - "$DB" <<'PY'
import sqlite3, sys
c = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
q = lambda s: c.execute(s).fetchone()[0]
print(f"  keyframes            : {q('select count(*) from Node')}")
print(f"  with an image        : {q('select count(*) from Data where image is not null and length(image)>0')}")
print(f"  visual words         : {q('select count(*) from Word')}")
print(f"  visual features      : {q('select count(*) from Feature')}")
names = {0: 'neighbour', 1: 'LOOP CLOSURE', 2: 'local space', 3: 'local time',
         4: 'user', 5: 'virtual', 6: 'landmark', 8: 'pose prior', 9: 'gravity'}
for t, n in c.execute('select type, count(*) from Link group by type'):
    print(f"  link type {t} ({names.get(t, '?')}): {n}")
PY
else
  echo "!!! no database written: $DB"
fi
echo "log: $LOG"
