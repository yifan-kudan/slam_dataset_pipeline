#!/usr/bin/env bash
# Compile (once) and run the headless map builder for kd-localizer-v2.0.0 (.kdm v9 + .kdllm).
#
# Runs wherever a 2.0 kd_amr_libs install and PDAL 2.6.2 headers sit side by side -- the
# release mapping container (kd_localizer_docker `mapping`, KD defaults to its mount), or a
# kd_amr_libs dev container with KD pointed at its install:
#
#   $SDP/scripts/mapping/build_map_headless_v9.sh --input cloud.las --output dir/localizer_map.kdm
#   KD=$HOME/ros2_ws/install/kd_amr_libs $SDP/scripts/mapping/build_map_headless_v9.sh --help
#
# KD overrides the kd_amr_libs install prefix; BIN overrides the cached binary path. Links only
# the mapping / map-maintenance libraries, no rosidl typesupport, so no ROS environment is needed.
set -euo pipefail

SRC=$(dirname "$(readlink -f "$0")")/build_map_headless_v9.cpp
BIN=${BIN:-/tmp/build_map_headless_v9}
KD=${KD:-/opt/kd_localizer/install/kd_amr_libs}

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    echo "compiling $(basename "$SRC") against $KD ..." >&2
    mkdir -p "$(dirname "$BIN")"
    g++ -std=c++17 -O2 "$SRC" \
        -I"$KD/include" -I/usr/include/eigen3 -I/usr/include/pdal -I/usr/local/include \
        -L"$KD/lib" -lkd_map_maintenance -lkd_mapping -lkd_sensor -lkd_geometry -lkd_math \
        -lpdalcpp -lyaml-cpp \
        -Wl,-rpath,"$KD/lib" -fopenmp -o "$BIN"
fi

exec "$BIN" "$@"
