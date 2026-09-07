#!/usr/bin/env bash
# Evaluate v4 against both controlled v2 and controlled v3 captures.
set -euo pipefail

ISAAC_ENV=${ISAAC_ENV:-/home/typist/miniconda3/envs/isaaclab}
WORKSPACE=${WORKSPACE:-/home/typist/dataset/dom_simulation}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE/datasets/evis_benchmark_10}
PYTHON="$ISAAC_ENV/bin/python"
EVALUATOR="$WORKSPACE/isaac-sim-event-camera-plugin/scripts/evaluate_evis_versions.py"
AGGREGATOR="$WORKSPACE/isaac-sim-event-camera-plugin/scripts/aggregate_evis_reports.py"

episode_h5() {
  find "$1" -maxdepth 1 -type f -name '*.h5' | sort | head -n 1
}

evaluate_pair() {
  local baseline=$1
  local candidate=$2
  local pair="${baseline}_vs_${candidate}"
  local out_dir="$DATA_ROOT/metrics/$pair"
  mkdir -p "$out_dir"

  for seed in {1..10}; do
    local base_dir="$DATA_ROOT/$baseline/demo${seed}"
    local cand_dir="$DATA_ROOT/$candidate/demo${seed}"
    local base_episode
    local cand_episode
    base_episode=$(episode_h5 "$base_dir")
    cand_episode=$(episode_h5 "$cand_dir")
    echo "[$pair/demo$seed] evaluating"
    "$PYTHON" "$EVALUATOR" \
      --v2-events "$base_dir/events/env0_ep${seed}.h5" \
      --v2-episode "$base_episode" \
      --v3-events "$cand_dir/events/env0_ep${seed}.h5" \
      --v3-episode "$cand_episode" \
      --out-json "$out_dir/evis_demo${seed}.json" \
      --out-md "$out_dir/evis_demo${seed}.md"
  done

  "$PYTHON" "$AGGREGATOR" \
    --input-glob "$out_dir/evis_demo*.json" \
    --out-json "$DATA_ROOT/metrics/${pair}_10episodes.json" \
    --out-md "$DATA_ROOT/metrics/${pair}_10episodes.md"
}

evaluate_pair v2_balanced v4_hybrid
evaluate_pair v3_adaptive v4_hybrid

echo "benchmark metrics complete: $DATA_ROOT/metrics"
