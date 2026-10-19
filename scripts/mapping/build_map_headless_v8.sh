#!/usr/bin/env bash
# Compile (once) and run the headless localizer-map builder for `main` (.kdm v8).
#
# Runs NATIVELY on this Jazzy host against the main build at ~/kd_localizer_jazzy/install.
# That install was copied out of the devcontainer, but this wrapper links only against
# kd_mapping / kd_geometry / kd_math -- no rosidl typesupport -- so the container-vs-host
# fastcdr drift that breaks the ROS nodes does not apply here.
#
#   $SDP/scripts/mapping/build_map_headless_v8.sh --input cloud.las --output map.kdm --overwrite
#   $SDP/scripts/mapping/build_map_headless_v8.sh --help
#
# KD overrides the kd_amr_libs install prefix; BIN overrides the cached binary path.
set -euo pipefail

SRC=$(dirname "$(readlink -f "$0")")/build_map_headless_v8.cpp
BIN=${BIN:-/home/kudan/prj/kd_rerun_20260825/build_map_headless_v8}
KD=${KD:-/home/kudan/kd_localizer_jazzy/install/kd_amr_libs}

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    echo "compiling $(basename "$SRC") ..." >&2
    mkdir -p "$(dirname "$BIN")"
    g++ -std=c++17 -O2 "$SRC" \
        -I"$KD/include" -I/usr/include/eigen3 -I/usr/include/pdal \
        -L"$KD/lib" -lkd_mapping -lkd_geometry -lkd_math -lpdalcpp \
        -Wl,-rpath,"$KD/lib" -fopenmp -o "$BIN"
fi

exec "$BIN" "$@"
