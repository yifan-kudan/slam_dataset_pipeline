#!/bin/bash
# One GLIM (GPU) mapping run on a Hilti 2023 sequence, headless, from the host.
#
#   scripts/hilti/run_glim.sh site2_handheld_5 [playback_duration_s]
#   GLIM_CONFIG=hilti_headless_kf scripts/hilti/run_glim.sh site2_handheld_5
#
# Uses glim_rosbag, NOT glim_rosnode + `ros2 bag play`: glim_rosbag reads the bag with
# rosbag2_cpp and calls points_callback directly, so no DDS is involved and no cloud can
# be lost. That mattered here -- FAST-LIO was losing ~47 % of the clouds to a writer-side
# shared-memory drop, and that loss is what broke site2_handheld_5 (config/dds/
# fastdds_shm_big.xml fixes it for FAST-LIO; GLIM never had the problem). glim_rosbag also
# throttles itself on odometry workload (>5 frames behind), so it cannot outrun the GPU.
#
# auto_quit is required: without it glim_rosbag spins forever after the last message
# instead of saving. dump_path gets traj_imu.txt / traj_lidar.txt (TUM, globally
# optimised) plus one directory per submap. Score it with src/hilti/eval_glim.py.
set -eu
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
q() { python3 "$REPO/src/paths.py" "$1"; }

SEQ=$1
DUR=${2:-0}
IMAGE=${GLIM_IMAGE:-koide3/glim_ros2:jazzy_cuda13.1}
CFG=$REPO/config/glim/${GLIM_CONFIG:-hilti_headless}
ROOT=$(q datasets.hilti_2023.root)
DUMP=$(q datasets.hilti_2023.results)/glim/$SEQ
BAG_SUFFIX=$(python3 -c "import sys;sys.path.insert(0,'$REPO/src');from hilti import dataset;print(dataset.BAG_SUFFIX)")

[ -d "$CFG" ] || { echo "no GLIM config dir: $CFG" >&2; exit 1; }
mkdir -p "$DUMP"
EXTRA=""
[ "$DUR" != "0" ] && EXTRA="-p playback_duration:=$DUR"

# shellcheck disable=SC2086
docker run --rm --name "glim-$SEQ" --gpus all --ipc=host \
  -v "$CFG":/glim/config:ro \
  -v "$ROOT":/data:ro \
  -v "$DUMP":/dump \
  "$IMAGE" \
  ros2 run glim_ros glim_rosbag "/data/${SEQ}${BAG_SUFFIX}" --ros-args \
    -p config_path:=/glim/config -p auto_quit:=true -p dump_path:=/dump $EXTRA
