#!/bin/bash
# Convert ConSLAM sequences to ROS 2 bags that include the RGB camera.
#
#   bash convert_conslam_with_cam.sh            # all of seq2-5
#   bash convert_conslam_with_cam.sh 3 5        # only those sequences
#
# Serial on purpose: every source is 63-92 GB on the same external HDD, and two
# concurrent readers thrash the head instead of going twice as fast. Not `set -e`
# either -- one sequence failing must not skip the rest.
#
# Groups: lidar,imu,rgb -> clouds + IMU + images + one camera_info per image +
# the velodyne->pp_rgb extrinsic on /tf_static. NIR is left out (`--groups all`
# adds it, +~100 GB, and its LiDAR extrinsic has not been derived yet).
#
# RETRIES exist because the drive's USB bridge (JMicron JMS561U 152d:1561, on the
# uas driver, behind a hub) drops its link under sustained load: the kernel logs
# `uas_eh_device_reset_handler` + `timing out command, waited 180s` and the read or
# write fails with EIO. There are NO medium errors -- the platter is fine, the link
# is not. A retry lands in a different window and usually gets through; the real
# fix is `usb-storage.quirks=152d:1561:u` (falls back to BOT) and/or a direct port.
# atomic_bag means a failed attempt leaves no partial bag behind, so retrying is safe.
set -u

REPO=/home/kudan/prj/slam_dataset_pipeline
SRC_ROOT=/media/kudan/seagate-2tb/datasets/ConSLAM
OUT=/media/kudan/seagate-2tb/datasets/ConSLAM_bag_with_cam
CFG=$REPO/config/conslam/bag_conslam_fastlio.yaml
ATTEMPTS=${ATTEMPTS:-3}

mkdir -p "$OUT"
LOG=$OUT/convert.log

# seq4's recording kept its original download name; the others were renamed.
declare -A SRC=(
  [2]="seq2/seq2_recording.bag"
  [3]="seq3/seq3_recording.bag"
  [4]="seq4/Copy of recording.bag"
  [5]="seq5/seq5_recording.bag"
)

SEQS=("$@")
[ ${#SEQS[@]} -eq 0 ] && SEQS=(2 3 4 5)

{
  echo "### start $(date -Is)   seqs=${SEQS[*]}  groups=lidar,imu,rgb  attempts=$ATTEMPTS"
  df -h "$OUT" | tail -1
  for n in "${SEQS[@]}"; do
    src="$SRC_ROOT/${SRC[$n]}"
    dst="$OUT/seq${n}_cam_bag"
    for try in $(seq 1 "$ATTEMPTS"); do
      echo
      echo "=== seq$n try $try/$ATTEMPTS  $(date -Is)  <- $src"
      start=$SECONDS
      python3 "$REPO/src/rosbag1_to_ros2.py" "$src" -o "$dst" -c "$CFG" \
        --groups lidar,imu,rgb --force
      code=$?
      printf '=== seq%s try %s exit=%s in %dm%02ds  %s\n' \
        "$n" "$try" "$code" $(( (SECONDS-start)/60 )) $(( (SECONDS-start)%60 )) \
        "$(du -sh "$dst" 2>/dev/null | cut -f1)"
      df -h "$OUT" | tail -1
      [ "$code" -eq 0 ] && break
      # Let the bridge settle (and the drive spin back up) before trying again.
      echo "--- seq$n failed, waiting 120 s for the USB link to settle"
      sleep 120
    done
  done
  echo
  echo "### finished $(date -Is)"
  du -sh "$OUT"/*_cam_bag 2>/dev/null
} 2>&1 | tee -a "$LOG"
