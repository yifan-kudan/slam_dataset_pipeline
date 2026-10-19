#!/bin/bash
# Rebuild every report input from the run artefacts, in dependency order.
#
# Run this rather than the individual scripts: the figures read data.json, so regenerating
# one without the other silently mixes runs, and shrink.py rewrites the PNGs in place so it
# has to come last.
#
# Needs the analysis interpreter (numpy, matplotlib, PIL, rosbags) -- set PY to pick one;
# the default comes from analysis_python in config/paths.yaml.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
REPO=$(cd ../../.. && pwd)
PY=${PY:-$(python3 "$REPO/src/paths.py" analysis_python)}
R=$(python3 -c "import sys;sys.path.insert(0,'$REPO/src');from hilti import dataset;print(dataset.report_dir())")

echo "== prep";      "$PY" prep.py
echo "== overview";  "$PY" fig_overview.py
echo "== special";   "$PY" fig_special.py
echo "== per-seq";   "$PY" fig_seq.py
echo "== tables";    "$PY" gen_tables.py
echo "== seqblocks"; "$PY" gen_seqblocks.py
echo "== shrink";    "$PY" shrink.py "$R/*.png" | tail -1
echo "== assemble";  "$PY" assemble.py
echo "== audit";     "$PY" audit.py
