#!/bin/bash
# Run a KdLocalizer campaign on the HOST, strictly one run at a time: every run is a real-time
# measurement and must not share the machine with another.
#
#   campaign.sh <campaign dir>                      every run in runs.tsv that is not done
#   campaign.sh <campaign dir> warm_s2 coldgpu_s2   only these runs
#   KIND=warm campaign.sh <campaign dir>            only the runs of one kind
#   FORCE=1 campaign.sh <campaign dir> warm_s2      repeat runs that are already done
#   LIMIT=60 campaign.sh <campaign dir> warm_s2     first 60 s of each bag (smoke test, never marked done)
#
# Long campaigns: nohup campaign.sh <dir> >> <dir>/logs/campaign.log 2>&1 &
# To stop one: kill the campaign.sh process, then `docker stop` the kdloc-run-* container.
#
# The campaign directory holds compose.yaml (the localizer container), campaign.env and runs/
# (both from src/kdloc/make_runs.py). For each run: wait until its inputs are ready, take the
# machine lock, start tegrastats (when the host has it) and procmon.py, run run_one.sh in the
# container, stop the samplers, and mark the run done (runs/<name>/done) when it produced a pose
# log and reported nothing [INVALID]. Runs that are not done are tried again in the next pass
# (PASSES, default 3).
#
# Optional hooks, executable files in <campaign dir>/hooks/:
#   ready <name>              exit 0 when the run's inputs are in place (default check: the bag's
#                             metadata.yaml and the map exist). It may also prepare them.
#   contended <start> <end>   print anything else that used the machine between two epoch
#                             seconds; any output keeps the run from being marked done.
# The lock is <campaign dir>/.machine.lock (flock). Other jobs on the machine, such as bag copies
# or checksum runs, can take it to stay out of the measurements.
set -u
CAMP=$(cd "${1:?usage: campaign.sh <campaign dir> [run names]}" && pwd)
shift
# shellcheck disable=SC1091
source "$CAMP/campaign.env"
mkdir -p "$CAMP/logs"
cd "$CAMP" || exit 1
exec 9>"$CAMP/.machine.lock"
PASSES=${PASSES:-3}
[ -n "${LIMIT:-}" ] && PASSES=1

run_env() {   # <name> <key>
  (set +u; source "$CAMP/runs/$1/run.env"; eval "echo \"\${$2}\"")
}

inputs_ready() {   # <name>
  if [ -x "$CAMP/hooks/ready" ]; then
    "$CAMP/hooks/ready" "$1"
    return
  fi
  [ -f "$(run_env "$1" BAG)/metadata.yaml" ] && [ -f "$(run_env "$1" MAP)" ]
}

run_once() {   # <name>
  local n=$1 run=$CAMP/runs/$1 out=$CAMP/runs/$1/out rc start end teg="" mon contended
  flock 9
  rm -rf "$out"
  rm -f "$run/done"
  mkdir -p "$out"
  date +%z > "$out/host_utc_offset"
  echo "$(date +%T) ############ $n"
  start=$(date +%s)
  if command -v tegrastats > /dev/null; then
    tegrastats --interval 1000 --logfile "$out/tegra.log" &
    teg=$!
  fi
  python3 "$SDP/src/kdloc/procmon.py" "$out/proc.csv" &
  mon=$!
  docker compose run --rm -T --name "kdloc-run-$$" ${LIMIT:+-e LIMIT="$LIMIT"} "$COMPOSE_SERVICE" \
    bash "$SDP/scripts/kdloc/run_one.sh" "$run" > "$out/run.log" 2>&1
  rc=$?
  # shellcheck disable=SC2086
  kill $teg $mon 2>/dev/null
  # shellcheck disable=SC2086
  wait $teg $mon 2>/dev/null
  end=$(date +%s)
  flock -u 9
  grep -E "^\[|localizer active|relocalizer ready" "$out/run.log"
  contended=""
  [ -x "$CAMP/hooks/contended" ] && contended=$("$CAMP/hooks/contended" "$start" "$end" | paste -sd, -)
  if [ -n "$contended" ]; then
    echo "$(date +%T) CONTENDED $n: other work during the run ($contended); repeated in the next pass"
  elif [ $rc -ne 0 ] || [ ! -s "$out/pose.csv" ] || grep -q '^\[INVALID\]' "$out/run.log"; then
    echo "$(date +%T) RUN FAILED $n (exit status $rc)"
  elif [ -n "${LIMIT:-}" ]; then
    echo "$(date +%T) $n: smoke run with LIMIT=$LIMIT, not marked done"
  else
    touch "$run/done"
  fi
}

names=${*:-$(awk -F'\t' -v k="${KIND:-}" 'NR > 1 && (k == "" || $2 == k) {print $1}' "$CAMP/runs.tsv")}
for pass in $(seq 1 "$PASSES"); do
  echo "$(date +%T) ======== pass $pass"
  pending=0
  for n in $names; do
    [ -f "$CAMP/runs/$n/run.env" ] || { echo "$(date +%T) SKIP $n (no runs/$n/run.env)"; continue; }
    if [ -f "$CAMP/runs/$n/done" ] && { [ "${FORCE:-0}" != "1" ] || [ "$pass" -gt 1 ]; }; then
      continue
    fi
    pending=$((pending + 1))
    until inputs_ready "$n"; do
      echo "$(date +%T) $n waiting for its inputs"
      sleep 120
    done
    run_once "$n"
  done
  [ "$pending" -eq 0 ] && break
done
total=$(($(wc -l < "$CAMP/runs.tsv") - 1))
echo "$(date +%T) CAMPAIGN PASSES COMPLETE: $(find "$CAMP/runs" -maxdepth 2 -name done | wc -l) of $total runs done"
