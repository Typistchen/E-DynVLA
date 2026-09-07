#!/usr/bin/env bash
# Generate deterministic v2/v3/v4 EVIS captures from the same current code.
set -euo pipefail

ISAAC_ENV=${ISAAC_ENV:-/home/typist/miniconda3/envs/isaaclab}
WORKSPACE=${WORKSPACE:-/home/typist/dataset/dom_simulation}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE/datasets/evis_benchmark_10}
MODE=${1:-all}
START_SEED=${2:-1}
END_SEED=${3:-10}
PYTHON="$ISAAC_ENV/bin/python"
SIM_ROOT="$WORKSPACE/dynamic-vla"
PLUGIN_ROOT="$WORKSPACE/isaac-sim-event-camera-plugin"
VALIDATOR="$WORKSPACE/V2E_VLA/benchmark/validate_evis_capture.py"
FFMPEG=$($PYTHON -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')

export OMNI_KIT_ACCEPT_EULA=YES

case "$MODE" in
  all) MODES=(v2_balanced v3_adaptive v4_hybrid) ;;
  v2_balanced|v3_adaptive|v4_hybrid) MODES=("$MODE") ;;
  *) echo "usage: $0 [all|v2_balanced|v3_adaptive|v4_hybrid] [start_seed] [end_seed]" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_ROOT"
{
  echo "generated_at=$(date --iso-8601=seconds)"
  echo "workspace=$WORKSPACE"
  echo "dynamic_vla_commit=$(git -C "$WORKSPACE/dynamic-vla" rev-parse HEAD)"
  echo "evis_commit=$(git -C "$WORKSPACE/isaac-sim-event-camera-plugin" rev-parse HEAD)"
  echo "threshold=0.15"
  echo "event_source=hdr"
  echo "event_warp=4"
  echo "hybrid_gate_gain=0.25"
  echo "hybrid_support_radius=2"
  "$PYTHON" -c 'import h5py; print("h5py=" + h5py.__version__); print("hdf5=" + h5py.version.hdf5_version)'
} > "$OUTPUT_ROOT/manifest.txt"

for mode in "${MODES[@]}"; do
  extra_flags=()
  case "$mode" in
    v2_balanced) ;;
    v3_adaptive) extra_flags+=(--event_adaptive_warp) ;;
    v4_hybrid) extra_flags+=(--event_hybrid --event_hybrid_gate_gain 0.25 --event_hybrid_support_radius 2) ;;
  esac

  for ((seed=START_SEED; seed<=END_SEED; seed++)); do
    demo="demo${seed}"
    out="$OUTPUT_ROOT/$mode/$demo"
    event_h5="$out/events/env0_ep${seed}.h5"
    validation="$out/validation.json"
    mkdir -p "$out/events"

    if [[ -f "$event_h5" ]]; then
      if "$PYTHON" "$VALIDATOR" "$event_h5" --expected-mode "$mode" --out-json "$validation" >/dev/null 2>&1; then
        echo "[$mode/$demo] valid capture exists; skipping simulation"
      else
        echo "[$mode/$demo] existing capture is invalid; refusing to overwrite: $event_h5" >&2
        exit 1
      fi
    else
      echo "[$mode/$demo] simulating seed=$seed"
      (
        cd "$SIM_ROOT"
        "$PYTHON" simulations/simulate.py \
          --headless --enable_cameras --num_envs 1 -n 1 \
          --task pick --seed "$seed" --save --debug \
          --event_camera --event_warp 4 \
          "${extra_flags[@]}" \
          --event_source hdr --event_threshold 0.15 \
          --scene_dir "$WORKSPACE/tests/DOM-Test/scenes" \
          --object_dir "$WORKSPACE/objects/DOM-3D-Objects/objects" \
          -o "$out"
      ) > "$out/sim.log" 2>&1
      "$PYTHON" "$VALIDATOR" "$event_h5" --expected-mode "$mode" --out-json "$validation" >/dev/null
      echo "[$mode/$demo] capture validated"
    fi

    event_video="$out/events/${demo}_event.mp4"
    if [[ ! -f "$event_video" ]]; then
      "$PYTHON" "$PLUGIN_ROOT/scripts/visualize_event.py" \
        --dir "$out/events" --env 0 --eps "$seed" --fps 25 --interval_ms 10 \
        --height 360 --width 480 --cams wrist_cam,opst_cam,side_cam \
        --out "$event_video" > "$out/visevent.log" 2>&1
    fi

    rgb_video=$(find "$out" -maxdepth 1 -type f -name '*.mp4' | sort | head -n 1)
    comparison="$out/comparison_${demo}_RGB_vs_Event.mp4"
    if [[ -n "$rgb_video" && ! -f "$comparison" ]]; then
      "$FFMPEG" -y -i "$rgb_video" -i "$event_video" -filter_complex \
        "[0:v]setpts=PTS-STARTPTS,scale=1440:720,setpts=N/(25*TB)[a];[1:v]setpts=PTS-STARTPTS,scale=1440:360,setpts=N/(25*TB)[b];[a][b]vstack,format=yuv420p" \
        -r 25 "$comparison" > "$out/ffmpeg.log" 2>&1
    fi
  done
done

echo "benchmark captures complete: $OUTPUT_ROOT"
