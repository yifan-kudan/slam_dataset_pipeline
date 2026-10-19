#!/usr/bin/env bash
# Run a command inside the Humble kd_localizer release container, with everything the
# Hilti scripts need mounted and pointed at by environment variables.
#
#   scripts/hilti/in_container.sh /opt/sdp/scripts/hilti/run_seq.sh site2_robot_1
#   SERVICE=mapping RUN=rerun_r1.0 scripts/hilti/in_container.sh \
#       /opt/sdp/scripts/hilti/build_maps.sh
#   NAME=site1_handheld_1_in_site1_handheld_4 scripts/hilti/in_container.sh \
#       /opt/sdp/scripts/hilti/run_seq.sh site1_handheld_1
#   scripts/hilti/in_container.sh bash              # poke around
#
# The release artifacts are built against Humble and cannot run on this Jazzy host, which
# is what the container is for. compose.yaml lives with the container (its path is
# localizer_container.compose_dir) and does not know about this checkout, so the three
# mounts that matter are added here rather than there:
#
#   this checkout -> $SDP   (default /opt/sdp)   the scripts and configs being run
#   the bag tree  -> same path as on the host    so a path in a config resolves either side
#   the results   -> $W     (default /work)      where every artefact is written
#
# Everything comes from config/paths.yaml; nothing in this file is absolute.
# COMPOSE_DRY_RUN=1 checks the assembled command (service, mounts, env) without
# starting anything -- it stops at "interactive run is not supported".
set -Eeuo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
q() { python3 "$REPO/src/paths.py" "$1"; }

SERVICE=${SERVICE:-localizer}
COMPOSE_DIR=$(q localizer_container.compose_dir)
SDP_MOUNT=$(q localizer_container.repo_mount)
WORK_MOUNT=$(q localizer_container.work_mount)
ROOT=$(q datasets.hilti_2023.root)
RESULTS=$(q datasets.hilti_2023.results)

[ -f "$COMPOSE_DIR/compose.yaml" ] || {
  echo "no compose.yaml in $COMPOSE_DIR (localizer_container.compose_dir)" >&2; exit 1; }
mkdir -p "$RESULTS"

# The knobs the in-container scripts read, forwarded only when set: RUN picks which
# FAST-LIO run build_maps.sh takes its cloud from, NAME which config run_seq.sh loads and
# what it calls its artefacts (a cross-sequence run replays one bag against another
# sequence's map, so it must not overwrite the own-map ones). `docker compose run` does
# not inherit the caller's environment, so without these two the scripts would silently
# fall back to their defaults.
# RELOC/RELOC_DELAY turn on the shipped 3D-BBS relocalizer and say how long after bag
# play it may start (see run_seq.sh).
FORWARD=()
[ -n "${RUN:-}" ] && FORWARD+=(-e "RUN=$RUN")
[ -n "${NAME:-}" ] && FORWARD+=(-e "NAME=$NAME")
[ -n "${RELOC:-}" ] && FORWARD+=(-e "RELOC=$RELOC")
[ -n "${RELOC_DELAY:-}" ] && FORWARD+=(-e "RELOC_DELAY=$RELOC_DELAY")

# --name: compose.yaml pins container_name per service, which `run` cannot reuse.
exec docker compose ${COMPOSE_DRY_RUN:+--dry-run} -f "$COMPOSE_DIR/compose.yaml" run --rm \
  --name "kd-$SERVICE-$$" \
  -v "$REPO:$SDP_MOUNT:ro" \
  -v "$ROOT:$ROOT" \
  -v "$RESULTS:$WORK_MOUNT" \
  -e "SDP=$SDP_MOUNT" \
  -e "W=$WORK_MOUNT" \
  -e "HILTI_ROOT=$ROOT" \
  "${FORWARD[@]}" \
  "$SERVICE" "$@"
