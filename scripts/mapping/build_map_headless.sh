#!/bin/bash
# Compile (once per container) and run the headless localizer-map builder.
#
# Run inside the mapping container, which has g++, Eigen and PDAL 2.6.2:
#   $SDP/scripts/mapping/build_map_headless.sh --input cloud.las --output map.kdm [options]
#   $SDP/scripts/mapping/build_map_headless.sh --help
#
# The binary is cached at /tmp/build_map_headless for the life of the container.

set -euo pipefail

SRC=$(dirname "$(readlink -f "$0")")/build_map_headless.cpp
BIN=${BIN:-/tmp/build_map_headless}
KD=${KD_LIBS_PREFIX:-/opt/kd_localizer/install/kd_amr_libs}

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    echo "compiling $(basename "$SRC") ..." >&2
    g++ -std=c++17 -O2 "$SRC" \
        -I"$KD/include" -I/usr/include/eigen3 -I/usr/include/pdal \
        -L"$KD/lib" -lkd_mapping -lkd_geometry -lkd_math -lpdalcpp \
        -Wl,-rpath,"$KD/lib" -fopenmp -o "$BIN"
fi

exec "$BIN" "$@"
