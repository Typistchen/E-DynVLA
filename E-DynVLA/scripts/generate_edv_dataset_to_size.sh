#!/usr/bin/env bash
set -uo pipefail

TARGET_GIB=${1:-500}
GPU_A=${2:-cuda:2}
GPU_B=${3:-cuda:3}

SCRATCH_ROOT=/vepfs-cnbj438438cfe4f9/scratch/jiaqi
DOM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET_ROOT="${EDV_DATASET_ROOT:-$SCRATCH_ROOT/dataset/EDV}"
LOG_ROOT="$SCRATCH_ROOT/logs/edv_500g"
TEMP_ROOT="$SCRATCH_ROOT/cache/edv_tmp"
COUNTER_FILE="$DATASET_ROOT/.next_sample_index"
COUNTER_LOCK="$DATASET_ROOT/.next_sample_index.lock"
TARGET_BYTES=$((TARGET_GIB * 1024 * 1024 * 1024))
MIN_FREE_BYTES=$((100 * 1024 * 1024 * 1024))

mkdir -p "$DATASET_ROOT/success" "$DATASET_ROOT/failure" "$LOG_ROOT" "$TEMP_ROOT"

# The server root filesystem is small and /tmp may fill during long Isaac Sim
# runs. Keep Python/Kit temporary files on the large VEPFS scratch volume.
export TMPDIR="$TEMP_ROOT"
export TMP="$TEMP_ROOT"
export TEMP="$TEMP_ROOT"

dataset_bytes() {
  du -sb "$DATASET_ROOT/success" "$DATASET_ROOT/failure" 2>/dev/null \
    | awk '{sum += $1} END {print sum + 0}'
}

initialize_counter() {
  if [[ -s "$COUNTER_FILE" ]]; then
    return
  fi
  local maximum=-1
  local path name index
  for path in "$DATASET_ROOT"/success/sample_* "$DATASET_ROOT"/failure/sample_*; do
    [[ -d "$path" ]] || continue
    name=${path##*/}
    index=$((10#${name#sample_}))
    (( index > maximum )) && maximum=$index
  done
  printf '%s\n' "$((maximum + 1))" > "$COUNTER_FILE"
}

allocate_index() {
  local index
  exec 9>"$COUNTER_LOCK"
  flock -x 9
  index=$(<"$COUNTER_FILE")
  printf '%s\n' "$((index + 1))" > "$COUNTER_FILE"
  flock -u 9
  exec 9>&-
  printf '%s\n' "$index"
}

run_worker() {
  local device=$1
  local current available sample_index
  while true; do
    current=$(dataset_bytes)
    if (( current >= TARGET_BYTES )); then
      echo "[$(date --iso-8601=seconds)] target reached bytes=$current device=$device"
      break
    fi
    available=$(df -PB1 "$DATASET_ROOT" | awk 'NR == 2 {print $4}')
    if (( available < MIN_FREE_BYTES )); then
      echo "[$(date --iso-8601=seconds)] stopped: free space below 100 GiB"
      break
    fi

    sample_index=$(allocate_index)
    echo "[$(date --iso-8601=seconds)] start sample=$sample_index device=$device bytes=$current"
    if EDV_DATASET_ROOT="$DATASET_ROOT" EDV_SAMPLER=dom_stratified \
      EDV_SEED_BASE=42 bash "$DOM_ROOT/scripts/generate_edv_samples.sh" \
      "$sample_index" 1 "$device"; then
      current=$(dataset_bytes)
      echo "[$(date --iso-8601=seconds)] done sample=$sample_index device=$device bytes=$current"
    else
      echo "[$(date --iso-8601=seconds)] error sample=$sample_index device=$device; continuing"
    fi
  done
}

initialize_counter
echo "[$(date --iso-8601=seconds)] target=${TARGET_GIB}GiB gpus=$GPU_A,$GPU_B"
run_worker "$GPU_A" >> "$LOG_ROOT/worker_${GPU_A//:/_}.log" 2>&1 &
PID_A=$!
run_worker "$GPU_B" >> "$LOG_ROOT/worker_${GPU_B//:/_}.log" 2>&1 &
PID_B=$!
printf '%s\n' "$PID_A" > "$LOG_ROOT/worker_${GPU_A//:/_}.pid"
printf '%s\n' "$PID_B" > "$LOG_ROOT/worker_${GPU_B//:/_}.pid"

wait "$PID_A"
wait "$PID_B"

"$SCRATCH_ROOT/environment/isaaclab45_dom/bin/python" -c \
  "import sys; from pathlib import Path; sys.path.insert(0, '$DOM_ROOT/scripts'); from package_edv_lerobot_sample import write_dataset_info; write_dataset_info(Path('$DATASET_ROOT'))"
echo "[$(date --iso-8601=seconds)] batch complete bytes=$(dataset_bytes)"
