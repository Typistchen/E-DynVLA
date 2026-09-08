#!/usr/bin/env bash
set -euo pipefail

START_ROW=${1:-0}
COUNT=${2:-1}
DEVICE=${3:-cuda:2}

if (( START_ROW < 0 || COUNT < 1 )); then
  echo "Usage: $0 [start_row>=0] [count>=1] [cuda:N]" >&2
  exit 2
fi

SCRATCH_ROOT=/vepfs-cnbj438438cfe4f9/scratch/jiaqi
PYTHON_BIN="$SCRATCH_ROOT/environment/isaaclab45_dom/bin/python"
DOM_ROOT="$SCRATCH_ROOT/code/E-DynVLA/dynamic-vla"
EVENT_ROOT="$SCRATCH_ROOT/code/raw_event_generator"
ASSET_ROOT="$SCRATCH_ROOT/dataset/dom_assets"
CSV_PATH="$SCRATCH_ROOT/dataset/pick_initial_conditions.csv"
DATASET_ROOT="${EDV_DATASET_ROOT:-$SCRATCH_ROOT/dataset/EDV}"
GENERATOR_REVISION=edv-v3-aedat4

export XDG_CACHE_HOME="$SCRATCH_ROOT/environment/cache"
export PIP_CACHE_DIR="$SCRATCH_ROOT/environment/cache/pip"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$EVENT_ROOT:${PYTHONPATH:-}"

mkdir -p "$DATASET_ROOT"

for (( row=START_ROW; row<START_ROW+COUNT; row++ )); do
  sample_name=$(printf 'sample_%06d' "$row")
  if [[ -e "$DATASET_ROOT/$sample_name" ]]; then
    echo "Refusing to overwrite existing sample: $DATASET_ROOT/$sample_name" >&2
    exit 3
  fi

  staging_root=$(mktemp -d "$SCRATCH_ROOT/dataset/.edv_staging_${row}.XXXXXX")
  cleanup_staging() {
    case "$staging_root" in
      "$SCRATCH_ROOT"/dataset/.edv_staging_*)
        rm -rf -- "$staging_root"
        ;;
      *)
        echo "Refusing to remove unexpected staging path: $staging_root" >&2
        ;;
    esac
  }
  trap cleanup_staging EXIT

  echo "[EDV] generating CSV row $row -> $sample_name on $DEVICE"
  "$PYTHON_BIN" -u "$DOM_ROOT/scripts/run_pick_csv_event_demo.py" \
    --dynamic-vla-root "$DOM_ROOT" \
    --csv "$CSV_PATH" \
    --row "$row" \
    --scene-dir "$ASSET_ROOT/scenes" \
    --object-dir "$ASSET_ROOT/objects" \
    --output-dir "$staging_root" \
    --device "$DEVICE" \
    --event-source hdr \
    --event-threshold 0.15 \
    --event-warp 4

  "$PYTHON_BIN" -u "$DOM_ROOT/scripts/package_edv_lerobot_sample.py" \
    --staging-dir "$staging_root" \
    --dataset-root "$DATASET_ROOT" \
    --csv "$CSV_PATH" \
    --row "$row" \
    --dynamic-vla-root "$DOM_ROOT" \
    --event-code-root "$EVENT_ROOT" \
    --asset-root "$ASSET_ROOT" \
    --generator-revision "$GENERATOR_REVISION" \
    --event-threshold 0.15 \
    --event-warp 4 \
    --event-source hdr \
    --device "$DEVICE" \
    --sample-index "$row"

  cleanup_staging
  trap - EXIT
done

echo "[EDV] generated $COUNT sample(s) in $DATASET_ROOT"
