#!/usr/bin/env python3
"""Package one generated DOM + Event episode as a reproducible EDV sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py


CODE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--dynamic-vla-root", type=Path, required=True)
    parser.add_argument("--event-code-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--visualizer", type=Path, required=True)
    parser.add_argument("--generator-revision", default="unknown")
    parser.add_argument("--event-threshold", type=float, required=True)
    parser.add_argument("--event-warp", type=int, required=True)
    parser.add_argument("--event-source", choices=("hdr", "ldr"), required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_code_tree(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in CODE_SUFFIXES
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def read_csv_row(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            if row_index == index:
                return row
    raise IndexError(f"CSV has no data row {index}")


def one(paths, description: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {description}, found {len(paths)}: {paths}")
    return paths[0]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def relative_asset(path: Path, asset_root: Path) -> str:
    try:
        return path.resolve().relative_to(asset_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    staging = args.staging_dir.resolve()
    dataset = args.dataset_root.resolve()
    if dataset.exists() and any(dataset.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset: {dataset}")

    episode_h5 = one(staging.glob("pick_*.h5"), "DOM episode HDF5")
    episode_json = one(staging.glob("pick_*.json"), "DOM episode JSON")
    episode_mp4 = one(staging.glob("pick_*.mp4"), "DOM preview MP4")
    event_h5 = one((staging / "events").glob("env*_ep*.h5"), "event HDF5")
    # Upstream DOM appends a random four-hex run suffix. EDV uses a stable ID
    # because source episode_index is unique for our CSV-conditioned corpus.
    upstream_stem = episode_h5.stem
    stem = re.sub(r"_[0-9a-fA-F]{4}$", "", upstream_stem)
    source_row = read_csv_row(args.csv, args.row)
    episode_index = int(source_row["episode_index"])

    for subdir in ("episodes", "events", "previews"):
        (dataset / subdir).mkdir(parents=True, exist_ok=True)

    with h5py.File(episode_h5, "r+") as handle:
        if "observation_timestamp_s" not in handle:
            raise RuntimeError("Episode lacks observation_timestamp_s; regenerate with EDV timing patch")
        frame_count = int(handle["observation_timestamp_s"].shape[0])
        obs_start = float(handle["observation_timestamp_s"][0])
        obs_end = float(handle["observation_timestamp_s"][-1])
        rgb_keys = sorted(key for key in handle if key.endswith("_rgb"))
        if not rgb_keys:
            raise RuntimeError("Episode has no RGB observations")
        height, width = map(int, handle[rgb_keys[0]].shape[1:3])
        handle.attrs["edv_schema_version"] = "1.0"
        handle.attrs["episode_id"] = stem
        handle.attrs["source_csv_name"] = args.csv.name
        handle.attrs["source_csv_row"] = args.row
        handle.attrs["source_csv_sha256"] = sha256_file(args.csv)

    with h5py.File(event_h5, "r+") as handle:
        if "DVS" not in handle:
            raise RuntimeError("Event file lacks /DVS group")
        cameras = sorted(handle["DVS"].keys())
        event_counts = {camera: int(handle["DVS"][camera]["t"].shape[0]) for camera in cameras}
        event_starts = [float(handle["DVS"][camera]["t"][0]) for camera in cameras if event_counts[camera]]
        event_ends = [float(handle["DVS"][camera]["t"][-1]) for camera in cameras if event_counts[camera]]
        handle.attrs["edv_schema_version"] = "1.0"
        handle.attrs["episode_id"] = stem
        handle.attrs["source_csv_name"] = args.csv.name
        handle.attrs["source_csv_row"] = args.row
        handle.attrs["source_csv_sha256"] = sha256_file(args.csv)
        handle.attrs["source_episode_index"] = episode_index
        handle.attrs["time_unit"] = "second"
        handle.attrs["timestamp_convention"] = "absolute_isaac_sim_time"
        handle.attrs["polarity_encoding"] = "+1_ON_-1_OFF"
        handle.attrs["confidence_range"] = "[0,1]"
        handle.attrs["camera_width"] = width
        handle.attrs["camera_height"] = height
        handle.attrs["evis_event_threshold"] = args.event_threshold
        handle.attrs["evis_requested_warp_steps"] = args.event_warp
        handle.attrs["evis_event_source"] = args.event_source

    event_preview = staging / "events" / f"{stem}.events.mp4"
    subprocess.run(
        [
            sys.executable,
            str(args.visualizer),
            "--dir",
            str(staging / "events"),
            "--env",
            "0",
            "--eps",
            str(episode_index),
            "--fps",
            "25",
            "--interval_ms",
            "10",
            "--height",
            str(height),
            "--width",
            str(width),
            "--cams",
            ",".join(cameras),
            "--out",
            str(event_preview),
        ],
        check=True,
    )

    with episode_json.open(encoding="utf-8") as stream:
        episode_metadata = json.load(stream)
    selected_assets = {}
    for key in ("house", "object", "container"):
        usd = episode_metadata.get("scene", {}).get(key, {}).get("spawn", {}).get("usd_path")
        if usd and Path(usd).is_file():
            asset_path = Path(usd)
            selected_assets[key] = {
                "path": relative_asset(asset_path, args.asset_root),
                "sha256": sha256_file(asset_path),
            }
    episode_metadata["edv_reproducibility"] = {
        "schema_version": "1.0",
        "upstream_episode_name": upstream_stem,
        "source_csv": {
            "name": args.csv.name,
            "sha256": sha256_file(args.csv),
            "zero_based_row": args.row,
            "row": source_row,
            "fields_used_by_generator": [
                "episode_index",
                "objects",
                "obj_pos_x",
                "obj_pos_y",
                "obj_pos_z",
                "obj_rot_x",
                "obj_rot_y",
                "obj_rot_z",
                "obj_vel_x",
                "obj_vel_y",
                "obj_vel_z",
            ],
        },
        "selected_assets": selected_assets,
        "generator_revision": args.generator_revision,
        "device": args.device,
        "event_configuration": {
            "mode": "v4_hybrid",
            "source": args.event_source,
            "threshold": args.event_threshold,
            "requested_warp_steps": args.event_warp,
            "adaptive_warp": True,
            "max_warp_factor": 2,
            "hybrid_gate_gain": 0.25,
            "hybrid_support_radius_px": 2,
            "composite": "log_blend",
            "motion_vector_dilation_px": 1,
        },
    }

    target_episode_h5 = dataset / "episodes" / f"{stem}.h5"
    target_episode_json = dataset / "episodes" / f"{stem}.json"
    target_event_h5 = dataset / "events" / f"{stem}.events.h5"
    target_preview = dataset / "previews" / f"{stem}.mp4"
    target_event_preview = dataset / "previews" / f"{stem}.events.mp4"
    shutil.copy2(episode_h5, target_episode_h5)
    with target_episode_json.open("w", encoding="utf-8") as stream:
        json.dump(episode_metadata, stream, ensure_ascii=False, indent=2)
    shutil.copy2(event_h5, target_event_h5)
    shutil.copy2(episode_mp4, target_preview)
    shutil.copy2(event_preview, target_event_preview)

    manifest_fields = [
        "episode_id",
        "task",
        "seed",
        "source_csv_row",
        "source_episode_index",
        "frames",
        "observation_start_s",
        "observation_end_s",
        "event_start_s",
        "event_end_s",
        "event_counts_json",
        "episode_h5",
        "event_h5",
        "preview_mp4",
        "event_preview_mp4",
        "episode_h5_sha256",
        "event_h5_sha256",
    ]
    manifest_row = {
        "episode_id": stem,
        "task": "pick",
        "seed": episode_metadata["seed"],
        "source_csv_row": args.row,
        "source_episode_index": episode_index,
        "frames": frame_count,
        "observation_start_s": f"{obs_start:.9f}",
        "observation_end_s": f"{obs_end:.9f}",
        "event_start_s": f"{min(event_starts):.9f}",
        "event_end_s": f"{max(event_ends):.9f}",
        "event_counts_json": json.dumps(event_counts, sort_keys=True),
        "episode_h5": target_episode_h5.relative_to(dataset).as_posix(),
        "event_h5": target_event_h5.relative_to(dataset).as_posix(),
        "preview_mp4": target_preview.relative_to(dataset).as_posix(),
        "event_preview_mp4": target_event_preview.relative_to(dataset).as_posix(),
        "episode_h5_sha256": sha256_file(target_episode_h5),
        "event_h5_sha256": sha256_file(target_event_h5),
    }
    with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerow(manifest_row)

    dataset_metadata = {
        "schema_name": "EDV",
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": 1,
        "source": {
            "dataset": "DOM pick initial conditions",
            "csv_name": args.csv.name,
            "csv_sha256": sha256_file(args.csv),
        },
        "generator": {
            "revision": args.generator_revision,
            "dynamic_vla_code_sha256": sha256_code_tree(args.dynamic_vla_root),
            "event_code_sha256": sha256_code_tree(args.event_code_root),
        },
        "software": {
            "python": platform.python_version(),
            "isaac_sim": package_version("isaacsim"),
            "isaac_lab": package_version("isaaclab"),
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "h5py": package_version("h5py"),
        },
        "event_configuration": episode_metadata["edv_reproducibility"]["event_configuration"],
        "time_alignment": {
            "episode_dataset": "observation_timestamp_s",
            "event_dataset": "DVS/<camera>/t",
            "event_time_origin_attribute": "event_time_origin_s",
            "unit": "second",
        },
        "reproduction_command": (
            "bash /vepfs-cnbj438438cfe4f9/scratch/jiaqi/code/"
            "E-DynVLA/E-DynVLA/scripts/generate_edv_row0.sh"
        ),
        "determinism_note": (
            "The realized scene and selected USD checksums are recorded. GPU physics/rendering "
            "may not be bitwise identical across driver or Isaac versions."
        ),
    }
    with (dataset / "dataset_meta.json").open("w", encoding="utf-8") as stream:
        json.dump(dataset_metadata, stream, ensure_ascii=False, indent=2)

    print(f"[EDV] packaged {stem} -> {dataset}", flush=True)


if __name__ == "__main__":
    main()
