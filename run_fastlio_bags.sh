#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="/home/kudan/Downloads/datasets/ConPR" # Change this to your dataset root directory
BAG_GLOB="*_fastlio_bag" # Change this to your bag discovery glob
PLAY_RATE="1.0"
READ_AHEAD_QUEUE_SIZE="10000"
OPEN_RVIZ=1
RECORD_ODOM=0
EVAL_ODOM=0
RVIZ_CONFIG="/home/kudan/prj/FAST_LIO_ROS2/rviz/fastlio.rviz"
CONFIG_FILE="conpr/fast_lio_conpr.yaml" # Config relative to config-path. For ConSLAM use conslam/fast_lio_conslam.yaml
CONFIG_PATH_OVERRIDE="" # If set (via --config-path), overrides the default repo config/ dir
MAP_OUTPUT_DIR=""
ODOM_OUTPUT_DIR=""
FASTLIO_ODOM_TOPIC="/Odometry"
ROS_LOG_DIR=""
WAIT_AFTER_LAUNCH=8
WAIT_AFTER_PLAY=5
WAIT_AFTER_SAVE=2
SERVICE_TIMEOUT=60

ROS_SETUP="/opt/ros/jazzy/setup.bash"
LIVOX_SETUP="/home/kudan/prj/ws_livox/install/setup.bash"
FASTLIO_SETUP="/home/kudan/prj/fast_lio_ws/install/setup.bash"

usage() {
  cat <<'EOF'
Usage:
  run_fastlio_bags.sh [options] [bag_dir ...]

Options:
  --root DIR          Dataset root directory. Default: /home/kudan/Downloads/datasets/ConPR
  --glob PATTERN      Auto-discovery glob under root. Default: *_fastlio_bag
  --config-file NAME  FAST-LIO config, relative to config-path. Default: conpr/fast_lio_conpr.yaml
  --config-path DIR   Directory holding the FAST-LIO config. Default: <repo>/config
  -r, --rate RATE     ros2 bag play rate. Default: 1.0
  --queue-size N      ros2 bag play --read-ahead-queue-size. Default: 10000
  --no-rviz           Do not open RViz
  --record-odom       Record FAST-LIO odometry topic while playing each bag
  --eval-odom         Export recorded odometry and evaluate against SEQUENCE/pose.csv
  --odom-topic TOPIC  FAST-LIO odometry topic. Default: /Odometry
  --odom-dir DIR      Directory where per-bag odometry bags/results are saved
  --ros-log-dir DIR   ROS log directory. Default: ROOT/ros_logs
  --rviz CONFIG       RViz config path
  --map-dir DIR       Directory where per-bag maps are copied
  -h, --help          Show this help

If bag_dir arguments are provided, auto-discovery is skipped.
EOF
}

bags=()
require_arg() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "missing value for ${option}" >&2
    exit 2
  fi
}

wait_for_service() {
  local service_name="$1"
  local timeout_sec="$2"
  local start_time="$SECONDS"

  while (( SECONDS - start_time < timeout_sec )); do
    if ros2 service list 2>/dev/null | grep -Fxq "$service_name"; then
      return 0
    fi
    sleep 1
  done

  echo "timeout waiting for service: ${service_name}" >&2
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      require_arg "$1" "${2:-}"
      ROOT_DIR="$2"
      shift 2
      ;;
    --glob)
      require_arg "$1" "${2:-}"
      BAG_GLOB="$2"
      shift 2
      ;;
    --config-file)
      require_arg "$1" "${2:-}"
      CONFIG_FILE="$2"
      shift 2
      ;;
    --config-path)
      require_arg "$1" "${2:-}"
      CONFIG_PATH_OVERRIDE="$2"
      shift 2
      ;;
    -r|--rate)
      require_arg "$1" "${2:-}"
      PLAY_RATE="$2"
      shift 2
      ;;
    --queue-size)
      require_arg "$1" "${2:-}"
      READ_AHEAD_QUEUE_SIZE="$2"
      shift 2
      ;;
    --no-rviz)
      OPEN_RVIZ=0
      shift
      ;;
    --record-odom)
      RECORD_ODOM=1
      shift
      ;;
    --eval-odom)
      RECORD_ODOM=1
      EVAL_ODOM=1
      shift
      ;;
    --odom-topic)
      require_arg "$1" "${2:-}"
      FASTLIO_ODOM_TOPIC="$2"
      shift 2
      ;;
    --odom-dir)
      require_arg "$1" "${2:-}"
      ODOM_OUTPUT_DIR="$2"
      shift 2
      ;;
    --ros-log-dir)
      require_arg "$1" "${2:-}"
      ROS_LOG_DIR="$2"
      shift 2
      ;;
    --rviz)
      require_arg "$1" "${2:-}"
      RVIZ_CONFIG="$2"
      shift 2
      ;;
    --map-dir)
      require_arg "$1" "${2:-}"
      MAP_OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      bags+=("$1")
      shift
      ;;
  esac
done

CONFIG_PATH="${CONFIG_PATH_OVERRIDE:-${SCRIPT_DIR}/config}"
MAP_SOURCE="${ROOT_DIR}/maps/fastlio_map.pcd"
if [[ -z "$MAP_OUTPUT_DIR" ]]; then
  MAP_OUTPUT_DIR="${ROOT_DIR}/maps/fastlio_runs"
fi
if [[ -z "$ODOM_OUTPUT_DIR" ]]; then
  ODOM_OUTPUT_DIR="${ROOT_DIR}/results/fastlio_odom"
fi
if [[ -z "$ROS_LOG_DIR" ]]; then
  ROS_LOG_DIR="${ROOT_DIR}/ros_logs"
fi
export ROS_LOG_DIR

source_if_exists() {
  local setup_file="$1"
  if [[ -f "$setup_file" ]]; then
    # shellcheck source=/dev/null
    set +u
    source "$setup_file"
    set -u
  else
    echo "missing setup file: $setup_file" >&2
    exit 1
  fi
}

resolve_bags() {
  if [[ ${#bags[@]} -gt 0 ]]; then
    printf '%s\n' "${bags[@]}"
    return
  fi

  find "$ROOT_DIR" -maxdepth 1 -type d -name "$BAG_GLOB" | sort
}

cleanup() {
  local pids=("${ODOM_RECORD_PID:-}" "${FASTLIO_PID:-}" "${RVIZ_PID:-}")
  for pid in "${pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
on_signal() {
  trap - EXIT
  cleanup
  exit 130
}
trap cleanup EXIT
trap on_signal INT TERM

source_if_exists "$ROS_SETUP"
source_if_exists "$LIVOX_SETUP"
source_if_exists "$FASTLIO_SETUP"

mkdir -p "$MAP_OUTPUT_DIR"
mkdir -p "$ODOM_OUTPUT_DIR"
mkdir -p "$ROS_LOG_DIR"
mapfile -t bag_paths < <(resolve_bags)

if [[ ${#bag_paths[@]} -eq 0 ]]; then
  echo "no bags found under ${ROOT_DIR} with glob ${BAG_GLOB}" >&2
  exit 1
fi

echo "bags to process:"
printf '  %s\n' "${bag_paths[@]}"

RVIZ_PID=""
if [[ "$OPEN_RVIZ" -eq 1 ]]; then
  setsid rviz2 -d "$RVIZ_CONFIG" --ros-args -p use_sim_time:=true &
  RVIZ_PID=$!
  sleep 3
fi

for bag in "${bag_paths[@]}"; do
  if [[ ! -d "$bag" ]]; then
    bag="${ROOT_DIR}/${bag}"
  fi
  if [[ ! -d "$bag" ]]; then
    echo "skip missing bag: $bag" >&2
    continue
  fi

  bag_name="$(basename "$bag")"
  sequence_name="${bag_name%_fastlio_bag}"
  sequence_dir="${ROOT_DIR}/${sequence_name}"
  map_copy="${MAP_OUTPUT_DIR}/${sequence_name}_fastlio_map.pcd"
  odom_bag="${ODOM_OUTPUT_DIR}/${sequence_name}_fastlio_odom"
  odom_tum="${ODOM_OUTPUT_DIR}/${sequence_name}_fastlio_odom.tum"
  odom_eval="${ODOM_OUTPUT_DIR}/${sequence_name}_fastlio_vs_pose_eval.txt"

  echo
  echo "=== FAST-LIO map: ${bag_name} ==="

  rm -f "$MAP_SOURCE"

  setsid ros2 launch fast_lio mapping.launch.py \
    config_path:="$CONFIG_PATH" \
    config_file:="$CONFIG_FILE" \
    use_sim_time:=true \
    rviz:=false &
  FASTLIO_PID=$!

  sleep "$WAIT_AFTER_LAUNCH"
  if ! kill -0 "$FASTLIO_PID" 2>/dev/null; then
    echo "FAST-LIO launch exited before bag playback for ${bag_name}" >&2
    wait "$FASTLIO_PID" 2>/dev/null || true
    FASTLIO_PID=""
    continue
  fi

  ODOM_RECORD_PID=""
  if [[ "$RECORD_ODOM" -eq 1 ]]; then
    if [[ -e "$odom_bag" ]]; then
      odom_bag="${odom_bag}_$(date +%Y%m%d_%H%M%S)"
      odom_tum="${odom_bag}.tum"
      odom_eval="${odom_bag}_vs_pose_eval.txt"
    fi
    echo "recording odometry: ${FASTLIO_ODOM_TOPIC} -> ${odom_bag}"
    setsid ros2 bag record "$FASTLIO_ODOM_TOPIC" /tf /tf_static -o "$odom_bag" &
    ODOM_RECORD_PID=$!
    sleep 2
  fi

  ros2 bag play "$bag" --clock -r "$PLAY_RATE" --read-ahead-queue-size "$READ_AHEAD_QUEUE_SIZE"

  if [[ -n "${ODOM_RECORD_PID:-}" ]] && kill -0 "$ODOM_RECORD_PID" 2>/dev/null; then
    kill -- "-$ODOM_RECORD_PID" 2>/dev/null || kill "$ODOM_RECORD_PID" 2>/dev/null || true
    wait "$ODOM_RECORD_PID" 2>/dev/null || true
  fi
  ODOM_RECORD_PID=""

  echo "bag play complete, waiting ${WAIT_AFTER_PLAY}s before saving map"
  sleep "$WAIT_AFTER_PLAY"

  wait_for_service /map_save "$SERVICE_TIMEOUT"
  ros2 service call /map_save std_srvs/srv/Trigger {}
  sleep "$WAIT_AFTER_SAVE"

  if [[ -f "$MAP_SOURCE" ]]; then
    cp "$MAP_SOURCE" "$map_copy"
    echo "saved map copy: $map_copy"
  else
    echo "warning: map source not found after save: $MAP_SOURCE" >&2
  fi

  if [[ "$EVAL_ODOM" -eq 1 ]]; then
    if [[ -d "$odom_bag" && -f "${sequence_dir}/pose.csv" ]]; then
      /usr/bin/python3 "${ROOT_DIR}/process/export_odometry_tum.py" \
        "$odom_bag" \
        -t "$FASTLIO_ODOM_TOPIC" \
        -o "$odom_tum"
      /usr/bin/python3 "${ROOT_DIR}/process/evaluate_pose_csv_tum.py" \
        "${sequence_dir}/pose.csv" \
        "$odom_tum" \
        -o "$odom_eval"
      echo "saved odom trajectory: $odom_tum"
      echo "saved odom evaluation: $odom_eval"
    else
      echo "warning: cannot evaluate odom for ${sequence_name}; missing odom bag or pose.csv" >&2
    fi
  fi

  if kill -0 "$FASTLIO_PID" 2>/dev/null; then
    kill -- "-$FASTLIO_PID" 2>/dev/null || kill "$FASTLIO_PID" 2>/dev/null || true
    sleep 1
    kill -9 -- "-$FASTLIO_PID" 2>/dev/null || kill -9 "$FASTLIO_PID" 2>/dev/null || true
    wait "$FASTLIO_PID" 2>/dev/null || true
  fi
  FASTLIO_PID=""
done

echo
echo "done. maps are in: $MAP_OUTPUT_DIR"
