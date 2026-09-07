#!/usr/bin/env bash
# Generate ten v4-hybrid captures with geometry and separate wrist events.
set -euo pipefail

ISAAC_ENV=${ISAAC_ENV:-/home/typist/miniconda3/envs/isaaclab}
WORKSPACE=${WORKSPACE:-/home/typist/dataset/dom_simulation}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE/datasets/motion_separation_10}
START_SEED=${1:-1}
END_SEED=${2:-10}
PYTHON="$ISAAC_ENV/bin/python"
SIM_ROOT="$WORKSPACE/dynamic-vla"
PLUGIN_ROOT="$WORKSPACE/isaac-sim-event-camera-plugin"
VALIDATOR="$WORKSPACE/V2E_VLA/benchmark/validate_evis_capture.py"

export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$OUTPUT_ROOT"
{
  echo "generated_at=$(date --iso-8601=seconds)"
  echo "workspace=$WORKSPACE"
  echo "dynamic_vla_commit=$(git -C "$SIM_ROOT" rev-parse HEAD)"
  echo "evis_commit=$(git -C "$PLUGIN_ROOT" rev-parse HEAD)"
  echo "evis_mode=v4_hybrid"
  echo "event_source=hdr"
  echo "event_threshold=0.15"
  echo "event_warp=4"
  echo "motion_separation=lighting_suppressed_ego_motion_v2"
  echo "flow_quantile=0.95"
  echo "event_confidence_threshold=0.50"
} > "$OUTPUT_ROOT/manifest.txt"

for ((seed=START_SEED; seed<=END_SEED; seed++)); do
  demo="demo${seed}"
  out="$OUTPUT_ROOT/$demo"
  event_h5="$out/events/env0_ep${seed}.h5"
  mkdir -p "$out/events"
  episode=$(find "$out" -maxdepth 1 -type f -name '*.h5' | sort | head -n 1)

  capture_valid=false
  if [[ -f "$event_h5" && -n "$episode" ]]; then
    if "$PYTHON" "$VALIDATOR" "$event_h5" \
        --expected-mode v4_hybrid --out-json "$out/validation.json" >/dev/null \
      && "$PYTHON" -c 'import h5py,sys; f=h5py.File(sys.argv[1],"r"); required={"wrist_cam_depth_metric","wrist_cam_motion_vectors","wrist_cam_pose_w_ros","wrist_cam_intrinsics"}; assert required <= set(f.keys())' "$episode"; then
      capture_valid=true
    fi
  fi

  if [[ "$capture_valid" != true ]]; then
    if [[ -f "$event_h5" || -n "$episode" ]]; then
      echo "[$demo] partial/invalid capture exists; refusing to overwrite $out" >&2
      exit 1
    fi
    echo "[$demo] simulating seed=$seed with ECDM geometry"
    (
      cd "$SIM_ROOT"
      "$PYTHON" simulations/simulate.py \
        --headless --enable_cameras --num_envs 1 -n 1 \
        --task pick --seed "$seed" --save --debug \
        --event_camera --event_warp 4 \
        --event_hybrid --event_hybrid_gate_gain 0.25 \
        --event_hybrid_support_radius 2 --event_dynamic_gt \
        --event_source hdr --event_threshold 0.15 \
        --scene_dir "$WORKSPACE/tests/DOM-Test/scenes" \
        --object_dir "$WORKSPACE/objects/DOM-3D-Objects/objects" \
        -o "$out"
    ) > "$out/sim.log" 2>&1
    episode=$(find "$out" -maxdepth 1 -type f -name '*.h5' | sort | head -n 1)
    "$PYTHON" "$VALIDATOR" "$event_h5" \
      --expected-mode v4_hybrid --out-json "$out/validation.json" >/dev/null
  else
    echo "[$demo] validated geometry capture exists"
  fi

  separation="$out/motion_separation_v2"
  metrics="$separation/wrist_cam_motion_separation_metrics.json"
  if [[ ! -f "$metrics" ]]; then
    echo "[$demo] separating static/dynamic/illumination events"
    "$PYTHON" "$PLUGIN_ROOT/scripts/separate_dynamic_static_events.py" \
      --episode "$episode" --events "$event_h5" \
      --camera wrist_cam --out-dir "$separation" \
      --fps 25 --event-window-ms 10 --flow-quantile 0.95 \
      > "$out/motion_separation.log" 2>&1
  else
    echo "[$demo] motion-separation result exists"
  fi
done

echo "motion-separation captures complete: $OUTPUT_ROOT"
