#!/usr/bin/env bash
# Rebuild the whole re-run report from the run artefacts, in dependency order.
#
# Run this rather than the four scripts by hand: figures.py and tables.py both read
# data.json, so regenerating one without collect.py silently mixes runs, and assemble.py
# embeds whatever PNGs are on disk.
#
# Needs the analysis interpreter (numpy, matplotlib, scipy, rosbags) -- set PY to pick
# one; the default comes from analysis_python in config/paths.yaml.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
REPO=$(cd ../../.. && pwd)
PY=${PY:-$(python3 "$REPO/src/paths.py" analysis_python)}

echo "== collect";  "$PY" collect.py
echo "== figures";  "$PY" figures.py
echo "== tables";   "$PY" tables.py
echo "== assemble"; "$PY" assemble.py "$@"
