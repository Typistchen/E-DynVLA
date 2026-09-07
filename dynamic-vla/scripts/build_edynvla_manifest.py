#!/usr/bin/env python3
"""Build a reproducible DOM + separated-event episode manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def _event_info(path: Path, sensor: str) -> dict:
    import h5py

    with h5py.File(path, "r", libver="latest") as handle:
        group = handle[f"DVS/{sensor}"]
        timestamps = group["t"]
        count = int(timestamps.shape[0])
        return {
            "event_count": count,
            "event_start_s": float(timestamps[0]) if count else None,
            "event_end_s": float(timestamps[-1]) if count else None,
            "event_time_origin_s": float(
                handle.attrs.get("event_time_origin_s", timestamps[0] if count else 0.0)
            ),
            "algorithm": str(handle.attrs.get("algorithm", "unknown")),
            "schema_version": int(handle.attrs.get("schema_version", 0)),
        }


def _dom_info(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r", libver="latest") as handle:
        return {
            "frame_count": int(handle["action"].shape[0]),
            "action_dim": int(handle["action"].shape[-1]),
            "state_sources": [key for key in ("joints", "ee_pos", "ee_quat") if key in handle],
            "rgb_sources": [key for key in ("opst_cam_rgb", "side_cam_rgb", "wrist_cam_rgb") if key in handle],
        }


def build_manifest(root: Path, output: Path, sensor: str, fps: float) -> dict:
    episodes = []
    def demo_number(path: Path) -> int:
        match = re.search(r"(\d+)$", path.name)
        return int(match.group(1)) if match else 0

    for demo_dir in sorted(root.glob("demo*"), key=demo_number):
        event_files = sorted(
            (demo_dir / "motion_separation_v2").glob("*_motion_separated_events.h5")
        )
        dom_files = sorted(
            path
            for path in demo_dir.glob("*.h5")
            if path.parent == demo_dir and path.name != "events"
        )
        if len(event_files) != 1 or len(dom_files) != 1:
            raise RuntimeError(
                f"{demo_dir}: expected one DOM H5 and one separated-event H5"
            )
        metadata_path = dom_files[0].with_suffix(".json")
        instruction = {}
        if metadata_path.exists():
            instruction = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "instruction", {}
            )
        episode = {
            "name": demo_dir.name,
            "dom_h5": str(dom_files[0].relative_to(root)),
            "event_h5": str(event_files[0].relative_to(root)),
            "metadata_json": str(metadata_path.relative_to(root)),
            "instruction": instruction,
            **_dom_info(dom_files[0]),
            **_event_info(event_files[0], sensor),
        }
        episodes.append(episode)

    manifest = {
        "format": "edynvla-dom-event-manifest-v1",
        "dataset_root": "${EDYNVLA_DATA_ROOT}",
        "source_root_name": root.name,
        "sensor": sensor,
        "fps": fps,
        "model_inputs": [
            "DOM RGB",
            "DOM robot state",
            "language instruction",
            "q_static event history",
            "q_dynamic event history",
        ],
        "evaluation_only": ["segmentation", "object_vel"],
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor", default="wrist_cam")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    manifest = build_manifest(args.root, args.output, args.sensor, args.fps)
    print(f"Wrote {len(manifest['episodes'])} episodes to {args.output}")


if __name__ == "__main__":
    main()
