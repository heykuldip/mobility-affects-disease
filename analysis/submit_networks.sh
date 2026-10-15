#!/bin/bash
# Build each run's contact network and its metrics, and the mobility measures, one Slurm array task per run
# (network.json, degrees.npy and mobility.json in each run folder).
#
#   bash analysis/submit_networks.sh <runs folder>              # lists the runs to do and shows the sbatch command
#   bash analysis/submit_networks.sh <runs folder> --submit
#
# <runs folder> is a folder made by scripts/run_sweep.sh (e.g. $OUT/sweep or $OUT/decay). Runs that already have
# network.json are skipped, so it can be rerun as runs finish. Each task runs analysis/networks.py on one run folder:
# about 1.5 hours to read a full log, plus the community and path-length calculations. Settings (environment):
# THROTTLE (tasks at once, default 10), MEM (default 200G: a full run peaked at 103 GB), TIME (default 0-08:00:00),
# PARTITION (default bigmem), EXCLUDE (nodes to avoid), EXTRA_ARGS (passed to networks.py, e.g. "--ncol"). The jobs
# inherit your shell's environment, which needs numpy, pandas, pyyaml and python-igraph. Filters: MODELS (model folders,
# e.g. MODELS=random_advan) and RUNS (run numbers, e.g. RUNS="1 2 3 4 5") pick a subset. FORCE=1 ignores earlier
# lists whose job is still active; a run whose own job is running is still skipped, by its network.started marker.
set -euo pipefail
TREE=${1:-}; [[ -d "$TREE" ]] || { sed -n '2,15p' "$0"; exit 1; }
SUBMIT=${2:-}
python3 -c "import numpy, pandas, yaml, igraph" 2>/dev/null || { echo "This Python lacks numpy, pandas, pyyaml or python-igraph: activate the environment first." >&2; exit 1; }
REPO=$(pwd); [[ -f "$REPO/analysis/networks.py" ]] || { echo "Run this from the repository root." >&2; exit 1; }
# Every submission gets its own list file, never rewritten: an array task reads its line from the list when it
# starts, so rewriting a list while some of its tasks still wait would hand them the wrong runs. Lists live in
# network_lists/<time>.txt, with the job's ID beside them in <time>.jobid.
reg="$TREE/network_lists"; mkdir -p "$reg"
active_job() { command -v squeue >/dev/null 2>&1 || return 0; squeue -h -j "$1" 2>/dev/null | grep -q .; }
busy="$reg/.busy.$$"; cand="$reg/.candidates.$$"; : > "$busy"; : > "$cand"
trap 'rm -f "$busy" "$cand"' EXIT
for jf in "$reg"/*.jobid; do                              # runs in any list whose job is still queued or running
  [[ -e "$jf" && "${FORCE:-0}" != 1 ]] || continue
  active_job "$(cat "$jf")" && cat "${jf%.jobid}.txt" >> "$busy"
done
for log in "$TREE"/*/runs/run_*/agent_log0.csv; do
  run=$(dirname "$log")
  [[ "$(basename "$run")" =~ ^run_[0-9]+$ ]] || continue     # skip earlier results moved aside (run_NN.previous_*)
  model=$(basename "$(dirname "$(dirname "$run")")"); num=$((10#${run##*run_}))
  if [[ -n "${MODELS:-}" && " $MODELS " != *" $model "* ]]; then continue; fi
  if [[ -n "${RUNS:-}" ]] && ! printf " %s " ${RUNS} | grep -q " 0*$num "; then continue; fi
  [[ -f "$run/run_info.json" ]] || continue          # the run hasn't finished
  [[ -f "$run/network.json" ]] && continue           # already done
  grep -qxF "$run" "$busy" && continue                # in a list whose job is still active
  if [[ -f "$run/network.started" ]]; then            # in progress, unless its job has left the queue
    jid=$(head -n 1 "$run/network.started")
    if ! command -v squeue >/dev/null 2>&1 || squeue -h -j "$jid" 2>/dev/null | grep -q .; then continue; fi
    echo "retrying $run (its earlier job $jid ended without network.json)"
  fi
  echo "$run" >> "$cand"
done
n=$(wc -l < "$cand")
list="$reg/$(date +%Y%m%d-%H%M%S).txt"
if [[ "$SUBMIT" == --submit && $n -gt 0 ]]; then cp "$cand" "$list"; fi
echo "$n runs to process$([[ "$SUBMIT" == --submit && $n -gt 0 ]] && echo " (list: $list)")"
[[ "$SUBMIT" == --submit ]] || sed "s#^#  #" "$cand"
[[ $n -gt 0 ]] || exit 0
mkdir -p "$TREE/network_logs"
cmd=(sbatch -J networks -p "${PARTITION:-bigmem}" -N 1 -n 1 -c 2 --mem="${MEM:-200G}" -t "${TIME:-0-08:00:00}"
     --array="1-$n%${THROTTLE:-10}" -o "$TREE/network_logs/%A_%a.out"
     --wrap "python3 $REPO/analysis/networks.py \"\$(sed -n \${SLURM_ARRAY_TASK_ID}p $list)\" ${EXTRA_ARGS:-}")
[[ -n "${EXCLUDE:-}" ]] && cmd+=(--exclude="$EXCLUDE")
if [[ "$SUBMIT" == --submit ]]; then
  jid=$("${cmd[@]}" --parsable | cut -d";" -f1)
  echo "$jid" > "${list%.txt}.jobid"
  echo "Submitted batch job $jid"
else
  echo "would run: ${cmd[*]}"
fi
