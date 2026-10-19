#!/bin/bash
# FAST-LIO cloud -> localizer .kdm, one map per Hilti 2023 sequence. Runs INSIDE the
# mapping container (it has PDAL, which the runtime image does not):
#
#   SERVICE=mapping scripts/hilti/in_container.sh $SDP/scripts/hilti/build_maps.sh
#   SERVICE=mapping scripts/hilti/in_container.sh $SDP/scripts/hilti/build_maps.sh site2_robot_1
#
# Which run each sequence's cloud comes from is config/hilti/sequences.yaml, so a map and
# the pos_log it shares a frame with cannot drift apart. Pass sequence names to limit the
# batch; RUN=<name> overrides the run for all of them (a sweep, e.g. RUN=qosfix_r0.3).
#
# Library defaults throughout -- 0.25 m plane voxels, 0.01 m plane-fit sigma, BBS pyramid
# at 2x the voxel size over 7 levels. The 0.01 default is not a compromise: on ConSLAM,
# loosening it to 0.05 raised the plane-voxel share from 13 % to 48 % and *dropped* the
# peak ray inlier ratio from 0.93 to 0.42, because a looser fit admits voxels whose
# "plane" is a smear.
set -u
SDP=${SDP:-/opt/sdp}
W=${W:-/work}
ROOT=${HILTI_ROOT:?set HILTI_ROOT to the bag tree (in_container.sh does)}
KDM=$(python3 "$SDP/src/paths.py" datasets.hilti_2023.kdm)
SEQS=${*:-$(python3 - "$SDP" <<'PY'
import sys; sys.path.insert(0, f"{sys.argv[1]}/src")
from hilti import dataset
print(" ".join(s for s, e in dataset.sequences().items() if e["kdm"]))
PY
)}
mkdir -p "$KDM"

for seq in $SEQS; do
  run=${RUN:-$(python3 - "$SDP" "$seq" <<'PY'
import sys; sys.path.insert(0, f"{sys.argv[1]}/src")
from hilti import dataset
print(dataset.sequence(sys.argv[2])["run"])
PY
)}
  bag=$ROOT/${seq}_fastlio_bag
  pcd=$bag/runs/$run/maps/fastlio_map.pcd
  [ -f "$pcd" ] || pcd=$bag/runs/$run/maps/${seq}_fastlio_map.pcd
  las=${pcd%.pcd}_aligned.las
  kdm=$KDM/${seq}_fastlio_map.kdm
  if [ ! -f "$pcd" ]; then echo "SKIP $seq (no cloud in $bag/runs/$run/maps)"; continue; fi
  if [ ! -f "$las" ]; then
    # Gravity-align the cloud on the way out. kd_localizer seeds its ESKF gravity at the
    # MAP frame's -Z (imu.cpp:139), and a FAST-LIO map frame inherits the IMU's mounting
    # -- 88-92 deg off on the Hilti robot, 177-180 deg on the handheld rig. The localizer
    # diverges in an otherwise perfect map without this. See src/hilti/align_map.py, which
    # writes the matrix read here (the mapping image has no numpy, so it is not computed
    # in this container).
    M=$(python3 -c "import json;print(json.load(open('$W/align_$seq.json'))['pdal_matrix'])" 2>/dev/null)
    [ -z "$M" ] && { echo "FAIL align $seq (run src/hilti/align_map.py on the host first)"; continue; }
    # 1 mm quantisation: LAS stores integers and PDAL's 1 cm default is coarse next to a
    # 0.25 m plane voxel fitted to 0.01 m.
    pdal translate "$pcd" "$las" -f transformation \
      --filters.transformation.matrix="$M" \
      --writers.las.scale_x=0.001 --writers.las.scale_y=0.001 --writers.las.scale_z=0.001 \
      --writers.las.offset_x=auto --writers.las.offset_y=auto --writers.las.offset_z=auto \
      || { echo "FAIL las $seq"; continue; }
  fi
  echo "=== $seq (run $run)"
  "$SDP/scripts/mapping/build_map_headless.sh" --input "$las" --output "$kdm" --overwrite \
    && echo "OK $seq -> $kdm ($(du -h "$kdm" | cut -f1))" || echo "FAIL kdm $seq"
done
