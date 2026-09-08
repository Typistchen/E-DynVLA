#!/usr/bin/env bash
set -euo pipefail

SCRATCH_ROOT=/vepfs-cnbj438438cfe4f9/scratch/jiaqi
PYTHON_BIN="$SCRATCH_ROOT/environment/isaaclab45_dom/bin/python"
DOM_ROOT="$SCRATCH_ROOT/code/E-DynVLA/dynamic-vla"
EVENT_ROOT="$SCRATCH_ROOT/code/raw_event_generator"
ASSET_ROOT="$SCRATCH_ROOT/dataset/dom_assets"
CSV_PATH="$SCRATCH_ROOT/dataset/pick_initial_conditions.csv"
DATASET_ROOT="${EDV_DATASET_ROOT:-$SCRATCH_ROOT/dataset/EDV}"
GENERATOR_REVISION=edv-v1

export XDG_CACHE_HOME="$SCRATCH_ROOT/environment/cache"
export PIP_CACHE_DIR="$SCRATCH_ROOT/environment/cache/pip"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$EVENT_ROOT:${PYTHONPATH:-}"

if [[ -d "$DATASET_ROOT" ]] && find "$DATASET_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty EDV dataset: $DATASET_ROOT" >&2
  echo "Set EDV_DATASET_ROOT to a new directory when reproducing." >&2
  exit 2
fi

STAGING_ROOT=$(mktemp -d "$SCRATCH_ROOT/dataset/.edv_row0_staging.XXXXXX")
cleanup() {
  case "$STAGING_ROOT" in
    "$SCRATCH_ROOT"/dataset/.edv_row0_staging.*)
      rm -rf -- "$STAGING_ROOT"
      ;;
    *)
      echo "Refusing to remove unexpected staging path: $STAGING_ROOT" >&2
      ;;
  esac
}
trap cleanup EXIT

"$PYTHON_BIN" -u "$DOM_ROOT/scripts/run_pick_csv_event_demo.py" \
  --dynamic-vla-root "$DOM_ROOT" \
  --csv "$CSV_PATH" \
  --row 0 \
  --scene-dir "$ASSET_ROOT/scenes" \
  --object-dir "$ASSET_ROOT/objects" \
  --output-dir "$STAGING_ROOT" \
  --device cuda:2 \
  --event-source hdr \
  --event-threshold 0.15 \
  --event-warp 4

"$PYTHON_BIN" -u "$DOM_ROOT/scripts/package_edv_sample.py" \
  --staging-dir "$STAGING_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --csv "$CSV_PATH" \
  --row 0 \
  --dynamic-vla-root "$DOM_ROOT" \
  --event-code-root "$EVENT_ROOT" \
  --asset-root "$ASSET_ROOT" \
  --visualizer "$EVENT_ROOT/scripts/visualize_event.py" \
  --generator-revision "$GENERATOR_REVISION" \
  --event-threshold 0.15 \
  --event-warp 4 \
  --event-source hdr \
  --device cuda:2

echo "EDV sample ready: $DATASET_ROOT"
