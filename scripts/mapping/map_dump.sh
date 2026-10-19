#!/bin/bash
# Compile (once per container) and run the headless .kdm dumper.
#
# Run inside the mapping container:
#   $SDP/scripts/mapping/map_dump.sh map.kdm --planes-only > planes.csv

set -euo pipefail

SRC=$(dirname "$(readlink -f "$0")")/map_dump.cpp
BIN=${BIN:-/tmp/map_dump}
KD=${KD_LIBS_PREFIX:-/opt/kd_localizer/install/kd_amr_libs}

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    echo "compiling $(basename "$SRC") ..." >&2
    g++ -std=c++17 -O2 "$SRC" \
        -I"$KD/include" -I/usr/include/eigen3 \
        -L"$KD/lib" -lkd_mapping -lkd_geometry -lkd_math \
        -Wl,-rpath,"$KD/lib" -fopenmp -o "$BIN"
fi

exec "$BIN" "$@"
