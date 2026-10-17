#!/bin/bash
# Build and submit the simulation runs on a Slurm cluster, from this repository (model scripts and configs) plus each
# model's input files.
#
#   bash scripts/run_sweep.sh sweep [--submit] [model ...]     30 runs per model, base seeds 1-30
#   bash scripts/run_sweep.sh decay [--submit] [variant ...]   NHTS Distance, one run per distance-decay variant (Fig. 3):
#                                                            exp (exponent 2), exp_3, exp_4 and exp_5
#
# Without --submit it builds the folders and prints the sbatch commands. Run from the repository root with the model
# files committed, so each run records the commit it ran. Results land in $OUT/<sweep|decay>/<model>/runs/run_NN/.
# Settings (environment): INPUTS (required: the folder holding each model's input folder), DECAY_INPUTS (the
# decay variants' input files), OUT (default /scratch/$USER/abm), IMG (container image, default
# ~/containers/abm-1.1.0.sif), EXCLUDE (nodes to avoid, comma-separated), ONLY (submit just these runs, e.g. ONLY=1 or
# ONLY=1,5-7; an earlier result is moved aside), THROTTLE (runs at once per model, default 10), STOP_AT (override the run
# length, for quick tests), PARTITION and MEMORY (override every model's partition and memory).
set -euo pipefail
MODE=${1:-}; shift || true
[[ "$MODE" =~ ^(sweep|decay)$ ]] || { sed -n '2,15p' "$0"; exit 1; }
SUBMIT=0; MODELS=()
for a in "$@"; do if [[ "$a" == --submit ]]; then SUBMIT=1; else MODELS+=("$a"); fi; done
if [[ ${#MODELS[@]} -eq 0 ]]; then
  if [[ "$MODE" == decay ]]; then MODELS=(exp exp_3 exp_4 exp_5)
  else MODELS=(nhts_advan nhts_distance nhts_random random_advan random_places); fi
fi

INPUTS=${INPUTS:?set INPUTS to the folder holding the input folders}
DECAY_INPUTS=${DECAY_INPUTS:-$INPUTS/nhts_distance/distance_decay_input}
OUT=${OUT:-/scratch/$USER/abm}
IMG=${IMG:-$HOME/containers/abm-1.1.0.sif}
THROTTLE=${THROTTLE:-10}
EXCLUDE=${EXCLUDE:-}
[[ "${PARTITION:-}" != *,* ]] || { echo "PARTITION must name one partition." >&2; exit 1; }
REPO=$(pwd)
declare -A folder=( [nhts_advan]=nhts_advan [nhts_distance]=nhts_distance [nhts_random]=nhts_random [random_advan]=random_advan [random_places]=random_all )
declare -A part=( [nhts_advan]=bigmem [nhts_distance]=normal [nhts_random]=bigmem [random_advan]=bigmem [random_places]=normal )
declare -A mem=( [nhts_advan]=200G [nhts_distance]=200G [nhts_random]=200G [random_advan]=488G [random_places]=200G )

[[ -d .git ]] || { echo "Run this from the repository root." >&2; exit 1; }
FILES="nhts_advan.py nhts_distance.py nhts_random.py random_advan.py random_places.py"
git diff --quiet HEAD -- $FILES config human_agent places.py || { echo "Commit the model files first, so each run records its code version." >&2; exit 1; }
[[ -f "$IMG" ]] || { echo "No container image at $IMG (set IMG)." >&2; exit 1; }
command -v singularity >/dev/null 2>&1 || { echo "Run 'module load singularity' first: the jobs inherit this shell's environment." >&2; exit 1; }
VERSION=$(git rev-parse --short HEAD)
if [[ ! -f "$IMG.sha256" || "$IMG" -nt "$IMG.sha256" ]]; then sha256sum "$IMG" | cut -d" " -f1 > "$IMG.sha256" 2>/dev/null || echo unknown > "$IMG.sha256"; fi
IMG_SHA=$(cat "$IMG.sha256")
echo "image: $IMG (sha256 ${IMG_SHA:0:12}...); excluding nodes: ${EXCLUDE:-none}"

case "$MODE" in
  sweep) TREE=sweep; RUNS=1-30 ;;
  decay) TREE=decay; RUNS=1 ;;
esac
NPROC=64; TIME=0-08:00:00
RUNS=${ONLY:-$RUNS}

for m in "${MODELS[@]}"; do
  code=$m                                                # the model script; decay variants all run NHTS Distance
  if [[ "$MODE" == decay ]]; then
    code=nhts_distance
    case "$m" in exp|exp_3|exp_4|exp_5) sfx="_$m" ;; *) echo "unknown decay variant $m" >&2; exit 1 ;; esac
  fi
  [[ -n "${folder[$code]:-}" ]] || { echo "unknown model $code" >&2; exit 1; }
  dir="$OUT/$TREE/$m"
  if [[ ! -d "$dir/input" ]]; then                       # inputs copied once per tree; runs never change them
    mkdir -p "$dir" && cp -r "$INPUTS/${folder[$code]}/input" "$dir/input"
  fi
  mkdir -p "$dir/config" "$dir/human_agent" "$dir/runs"
  cp "$REPO/$code.py" "$dir/data_model.py"; cp "$REPO"/human_agent/*.py "$dir/human_agent/"; cp "$REPO/places.py" "$dir/"
  echo "$IMG_SHA" > "$dir/IMAGE_SHA256"
  cp "$REPO/config/$code.yaml" "$dir/config/base.yaml"; echo "$VERSION" > "$dir/CODE_VERSION"
  : > "$dir/config/extra.yaml"
  [[ -n "${STOP_AT:-}" ]] && echo "stop.at: $STOP_AT" >> "$dir/config/extra.yaml"
  if [[ "$MODE" == decay ]]; then                        # each input file set explicitly, by its own key
    names=("ffx_agents_trips_distance$sfx.csv" "dwell_time_distance$sfx.json" "restaurants_dict_distance$sfx.json"
           "recreation_dict_distance$sfx.json" "errands_dict_distance$sfx.json" "hospitals_dict_distance$sfx.json")
    keys=(input_agents input_dwell_time input_restaurants input_recreations input_errands input_hospitals)
    for i in "${!names[@]}"; do
      f=${names[$i]}
      [[ -f "$DECAY_INPUTS/$f" ]] || { echo "missing $DECAY_INPUTS/$f" >&2; exit 1; }
      cp "$DECAY_INPUTS/$f" "$dir/input/"
      echo "${keys[$i]}: 'input/$f'" >> "$dir/config/extra.yaml"
    done
  fi
  PART=${PARTITION:-${part[$code]}}; MEM=${MEMORY:-${mem[$code]}}
  cat > "$dir/job.sh" <<'JOB'
#!/bin/bash
# One run: its own folder and config (base seed = array index), then the model with the container's own mpiexec.
set -euo pipefail
cd "$MODEL_DIR"
n=$SLURM_ARRAY_TASK_ID; run=runs/run_$(printf %02d "$n")
if [[ -e "$run/agent_log0.csv" || -e "$run/run_info.json" ]]; then     # never write over an earlier result
  old="${run}.previous_$(date +%Y%m%d%H%M%S)"; mv "$run" "$old"; echo "earlier result moved to $old"
fi
mkdir -p "$run"
# drop from the base config the keys this run sets: seed, log, and anything in the extra file
keys='random\.seed|agent_log_file|write_agent_log'
extra=$( { grep -o -E '^[A-Za-z_.]+' config/extra.yaml || true; } | sed 's/\./\\./g' | paste -sd'|' -)
[[ -n "$extra" ]] && keys="$keys|$extra"
grep -v -E "^($keys):" config/base.yaml > "$run/config.yaml" || true
printf "\nrandom.seed: %s\nagent_log_file: '%s/agent_log'\nwrite_agent_log: true\n" "$n" "$run" >> "$run/config.yaml"
cat config/extra.yaml >> "$run/config.yaml"
cp CODE_VERSION "$run/"
# --cleanenv: the container's MPI launcher forwards the whole environment to every process and fails with too many
# variables, so only the image's own settings remain, plus these, passed through SINGULARITYENV_.
export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_MKL_NUM_THREADS=1
command -v singularity >/dev/null 2>&1 || module load singularity 2>/dev/null || true
command -v singularity >/dev/null 2>&1 || { echo "singularity not found: run 'module load singularity' before submitting"; exit 1; }
start=$(date -Is)
echo "$start $(hostname -s) run $n starting: singularity exec --cleanenv $IMG mpiexec -n $NPROC python ./data_model.py $run/config.yaml"
set +e
singularity exec --cleanenv "$IMG" mpiexec -n "$NPROC" python ./data_model.py "$run/config.yaml" > "$run/model.out" 2> "$run/model.err"
status=$?
set -e
end=$(date -Is)
# a manifest for every run, finished or not
printf '{\n "model_dir": "%s",\n "run": "%s",\n "base_seed": %s,\n "code_version": "%s",\n "image": "%s",\n "image_sha256": "%s",\n "slurm_job_id": "%s",\n "slurm_array_job_id": "%s",\n "slurm_array_task_id": "%s",\n "node": "%s",\n "processes": %s,\n "start": "%s",\n "end": "%s",\n "exit_status": %s\n}\n' \
  "$MODEL_DIR" "$run" "$n" "$(cat CODE_VERSION)" "$IMG" "$(cat IMAGE_SHA256 2>/dev/null || echo unknown)" "${SLURM_JOB_ID:-}" \
  "${SLURM_ARRAY_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID:-}" "$(hostname -s)" "$NPROC" "$start" "$end" "$status" > "$run/manifest.json"
if [[ $status -ne 0 ]]; then
  echo "$end run $n FAILED (exit $status). The end of $run/model.err:"
  tail -n 25 "$run/model.err"
  exit 1
fi
echo "$end run $n finished"
JOB
  chmod +x "$dir/job.sh"
  cmd=(sbatch -J "abm_$m" -p "$PART" -N 1 -n 1 -c "$NPROC" --mem="$MEM" -t "$TIME" --array="$RUNS%$THROTTLE"
       -o "$dir/runs/slurm-%A_%a.out" --export=ALL,MODEL_DIR="$dir",IMG="$IMG",NPROC="$NPROC")
  [[ -n "$EXCLUDE" ]] && cmd+=(--exclude="$EXCLUDE")
  cmd+=("$dir/job.sh")
  echo "$m: $dir  (code $VERSION; $PART, $MEM, $NPROC processes, runs $RUNS)"
  if [[ $SUBMIT == 1 ]]; then "${cmd[@]}"; else echo "    would run: ${cmd[*]}"; fi
done
