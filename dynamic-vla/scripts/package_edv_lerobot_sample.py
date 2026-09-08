#!/usr/bin/env python3
"""Package one DOM+Event run as one self-contained EDV sample."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dv_processing as dv
import h5py
import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


CAMERAS = ("wrist_cam", "opst_cam", "side_cam")
FPS = 25
WIDTH = 480
HEIGHT = 360
AEDAT_PACKET_EVENTS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--dynamic-vla-root", type=Path, required=True)
    parser.add_argument("--event-code-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--generator-revision", default="edv-v3-aedat4")
    parser.add_argument("--event-threshold", type=float, required=True)
    parser.add_argument("--event-warp", type=int, required=True)
    parser.add_argument("--event-source", choices=("hdr", "ldr"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_code_tree(root: Path) -> str:
    digest = hashlib.sha256()
    suffixes = {".py", ".sh", ".yaml", ".yml", ".toml"}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in suffixes
            or ".git" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def read_csv_row(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            if row_index == index:
                return row
    raise IndexError(f"CSV has no data row {index}")


def require_one(paths, label: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def quaternion_to_euler(quaternion_wxyz: np.ndarray) -> np.ndarray:
    xyzw = quaternion_wxyz[..., [1, 2, 3, 0]]
    euler = Rotation.from_quat(xyzw).as_euler("xyz").astype(np.float32)
    euler[..., [0, 2]] = np.mod(euler[..., [0, 2]], 2 * np.pi)
    return euler


def encode_video(frames: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        output_path,
        frames,
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    decoded_count = 0
    for decoded in iio.imiter(output_path):
        if decoded.shape != frames.shape[1:]:
            raise RuntimeError(
                f"Video resolution mismatch for {output_path}: {decoded.shape}"
            )
        decoded_count += 1
    if decoded_count != len(frames):
        raise RuntimeError(
            f"Video frame count mismatch for {output_path}: "
            f"{decoded_count} != {len(frames)}"
        )


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_aedat4_camera(
    source_path: Path,
    target_path: Path,
    camera: str,
    time_origin_s: float,
) -> dict[str, Any]:
    """Convert one HDF5 camera stream to standard x/y/t/p AEDAT4."""
    with h5py.File(source_path, "r", libver="latest") as source:
        group = source[f"DVS/{camera}"]
        x = group["x"][:].astype(np.int16, copy=False)
        y = group["y"][:].astype(np.int16, copy=False)
        timestamps_us = np.rint(
            (group["t"][:].astype(np.float64) - time_origin_s) * 1_000_000.0
        ).astype(np.int64)
        polarity = group["p"][:] > 0
        source_attrs = {
            str(key): jsonable(value) for key, value in source.attrs.items()
        }

    lengths = {len(x), len(y), len(timestamps_us), len(polarity)}
    if len(lengths) != 1:
        raise RuntimeError(f"Mismatched event arrays for {camera}: {lengths}")
    if len(timestamps_us) == 0:
        raise RuntimeError(f"No events recorded for {camera}")
    if timestamps_us.min() < 0:
        raise RuntimeError(
            f"{camera} contains events before the shared origin: {timestamps_us.min()} us"
        )
    if x.min() < 0 or x.max() >= WIDTH or y.min() < 0 or y.max() >= HEIGHT:
        raise RuntimeError(f"Out-of-range coordinates in {camera}")

    if np.any(timestamps_us[1:] < timestamps_us[:-1]):
        order = np.argsort(timestamps_us, kind="stable")
        x, y = x[order], y[order]
        timestamps_us, polarity = timestamps_us[order], polarity[order]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    config = dv.io.MonoCameraWriter.EventOnlyConfig(camera, (WIDTH, HEIGHT))
    writer = dv.io.MonoCameraWriter(str(target_path), config)
    writer.setPackagingCount(AEDAT_PACKET_EVENTS)
    for start in range(0, len(timestamps_us), AEDAT_PACKET_EVENTS):
        stop = min(start + AEDAT_PACKET_EVENTS, len(timestamps_us))
        event_objects = [
            dv.Event(int(t), int(xi), int(yi), bool(pi))
            for t, xi, yi, pi in zip(
                timestamps_us[start:stop],
                x[start:stop],
                y[start:stop],
                polarity[start:stop],
            )
        ]
        packet = dv.EventPacket(dv.EventPacket.EventVector(event_objects))
        writer.writeEventPacket(packet)
    del writer
    gc.collect()

    reader = dv.io.MonoCameraRecording(str(target_path))
    resolution = tuple(int(v) for v in reader.getEventResolution())
    count = 0
    lowest = None
    highest = None
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None or batch.isEmpty():
            continue
        count += int(batch.size())
        batch_low = int(batch.getLowestTime())
        batch_high = int(batch.getHighestTime())
        lowest = batch_low if lowest is None else min(lowest, batch_low)
        highest = batch_high if highest is None else max(highest, batch_high)
    if count != len(timestamps_us):
        raise RuntimeError(
            f"AEDAT4 round-trip count mismatch for {camera}: "
            f"{count} != {len(timestamps_us)}"
        )
    if resolution != (WIDTH, HEIGHT):
        raise RuntimeError(f"AEDAT4 resolution mismatch for {camera}: {resolution}")

    return {
        "event_count": count,
        "timestamp_min_us": lowest,
        "timestamp_max_us": highest,
        "size_bytes": target_path.stat().st_size,
        "source_hdf5_attrs": source_attrs,
    }


def write_dataset_info(root: Path) -> None:
    samples = []
    for reproduction_path in sorted(root.glob("sample_*/reproduction.json")):
        with reproduction_path.open(encoding="utf-8") as stream:
            reproduction = json.load(stream)
        samples.append(
            {
                "sample_index": reproduction["sample_index"],
                "source_csv_row": reproduction["source_csv"]["zero_based_row"],
                "source_episode_index": reproduction["source_csv"][
                    "source_episode_index"
                ],
                "frame_count": reproduction["frame_count"],
                "event_counts": {
                    camera: reproduction["events"][camera]["event_count"]
                    for camera in CAMERAS
                },
            }
        )
    info = {
        "schema_version": "edv-3.0",
        "description": "DOM RGB/action/state observations paired with v4-hybrid events",
        "sample_path": "sample_{sample_index:06d}",
        "cameras": list(CAMERAS),
        "resolution_wh": [WIDTH, HEIGHT],
        "observation_fps": FPS,
        "parquet_schema": [
            "action",
            "observation.state",
            "observation.environment_state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
        "event_format": {
            "container": "AEDAT4",
            "fields": ["x", "y", "t", "p"],
            "timestamp_unit": "microsecond",
            "timestamp_origin": "first observation timestamp",
            "polarity": "boolean (OFF=0, ON=1)",
            "confidence_q_saved": False,
            "observation_offsets_saved": False,
            "alignment": "derive later from AEDAT4 t and Parquet timestamp",
        },
        "samples": samples,
        "total_samples": len(samples),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    temp_path = root / ".dataset_info.json.tmp"
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(info, stream, ensure_ascii=False, indent=2)
    temp_path.replace(root / "dataset_info.json")


def main() -> None:
    args = parse_args()
    staging = args.staging_dir.resolve()
    root = args.dataset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sample_name = f"sample_{args.sample_index:06d}"
    final_sample = root / sample_name
    temp_sample = root / f".{sample_name}.tmp"
    if final_sample.exists():
        raise FileExistsError(f"Refusing to overwrite existing sample: {final_sample}")
    if temp_sample.exists():
        shutil.rmtree(temp_sample)
    temp_sample.mkdir()

    try:
        source_h5 = require_one(staging.glob("pick_*.h5"), "DOM episode HDF5")
        source_json = require_one(staging.glob("pick_*.json"), "DOM episode JSON")
        source_event = require_one(
            (staging / "events").glob("env*_ep*.h5"), "event HDF5"
        )
        csv_row = read_csv_row(args.csv, args.row)
        source_episode_index = int(csv_row["episode_index"])

        with source_json.open(encoding="utf-8") as stream:
            simulation_config = json.load(stream)

        with h5py.File(source_h5, "r") as source:
            required = {
                "action",
                "ee_pos",
                "ee_quat",
                "object_pos",
                "object_quat",
                "object_vel",
                "observation_timestamp_s",
                *(f"{camera}_rgb" for camera in CAMERAS),
            }
            missing = required - set(source.keys())
            if missing:
                raise RuntimeError(f"Source episode is missing: {sorted(missing)}")
            timestamps_abs = source["observation_timestamp_s"][:].astype(np.float64)
            frame_count = len(timestamps_abs)
            if frame_count == 0:
                raise RuntimeError("Source episode contains no observations")
            time_origin_s = float(timestamps_abs[0])
            timestamps_rel = timestamps_abs - time_origin_s
            action_raw = source["action"][:]
            actions = np.concatenate(
                (
                    action_raw[:, :3],
                    quaternion_to_euler(action_raw[:, 3:7]),
                    action_raw[:, -1:],
                ),
                axis=1,
            ).astype(np.float32)
            state = np.concatenate(
                (source["ee_pos"][:], quaternion_to_euler(source["ee_quat"][:])),
                axis=1,
            ).astype(np.float32)
            environment_state = np.concatenate(
                (
                    source["object_pos"][:],
                    quaternion_to_euler(source["object_quat"][:]),
                    source["object_vel"][:],
                ),
                axis=1,
            ).astype(np.float32)
            rgb_frames = {camera: source[f"{camera}_rgb"][:] for camera in CAMERAS}

        expected_timestamp = np.arange(frame_count, dtype=np.float64) / FPS
        if not np.allclose(timestamps_rel, expected_timestamp, atol=1e-6):
            raise RuntimeError(
                "Observation timestamps are not on the expected 25 Hz common timebase"
            )

        parquet_path = (
            temp_sample / "data" / f"episode_{args.sample_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(
            {
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
                "observation.environment_state": pa.array(
                    environment_state.tolist(), type=pa.list_(pa.float32())
                ),
                "timestamp": pa.array(timestamps_rel, type=pa.float64()),
                "frame_index": pa.array(np.arange(frame_count), type=pa.int64()),
                "episode_index": pa.array(
                    np.full(frame_count, args.sample_index), type=pa.int64()
                ),
                "index": pa.array(np.arange(frame_count), type=pa.int64()),
                "task_index": pa.array(
                    np.full(frame_count, args.sample_index), type=pa.int64()
                ),
            }
        )
        pq.write_table(table, parquet_path)
        reread = pq.read_table(parquet_path)
        if reread.schema != table.schema or reread.num_rows != frame_count:
            raise RuntimeError("Parquet round-trip validation failed")

        video_paths = {}
        for camera in CAMERAS:
            frames = rgb_frames[camera]
            if frames.shape != (frame_count, HEIGHT, WIDTH, 3):
                raise RuntimeError(f"Unexpected RGB shape for {camera}: {frames.shape}")
            video_path = temp_sample / "rgb" / f"{camera}.mp4"
            encode_video(frames, video_path)
            video_paths[camera] = video_path
        del rgb_frames
        gc.collect()

        event_paths = {}
        event_metadata = {}
        for camera in CAMERAS:
            event_path = temp_sample / "events" / f"{camera}.aedat4"
            event_metadata[camera] = write_aedat4_camera(
                source_event, event_path, camera, time_origin_s
            )
            event_paths[camera] = event_path

        selected_assets = {}
        for name in ("house", "object", "container"):
            path_string = (
                simulation_config.get("scene", {})
                .get(name, {})
                .get("spawn", {})
                .get("usd_path")
            )
            if path_string and Path(path_string).is_file():
                asset = Path(path_string)
                try:
                    relative = asset.resolve().relative_to(
                        args.asset_root.resolve()
                    ).as_posix()
                except ValueError:
                    relative = str(asset)
                selected_assets[name] = {
                    "path": relative,
                    "sha256": sha256_file(asset),
                }

        files = {
            parquet_path.relative_to(temp_sample).as_posix(): sha256_file(parquet_path)
        }
        for path in (*video_paths.values(), *event_paths.values()):
            files[path.relative_to(temp_sample).as_posix()] = sha256_file(path)

        reproduction = {
            "schema_version": "edv-3.0",
            "sample_index": args.sample_index,
            "frame_count": frame_count,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generator_revision": args.generator_revision,
            "source_csv": {
                "name": args.csv.name,
                "sha256": sha256_file(args.csv),
                "zero_based_row": args.row,
                "source_episode_index": source_episode_index,
                "source_task_index": int(csv_row["task_index"]),
                "row": csv_row,
            },
            "simulation_seed": source_episode_index,
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
                "stored_fields": ["x", "y", "t", "p"],
                "confidence_q_saved": False,
            },
            "time_alignment": {
                "common_origin_isaac_time_s": time_origin_s,
                "parquet_timestamp_s": "observation_timestamp_s - common_origin",
                "aedat4_timestamp_us": (
                    "round((event_t_s - common_origin) * 1e6)"
                ),
                "observation_offsets_saved": False,
                "later_alignment": (
                    "search AEDAT4 timestamps using Parquet timestamp * 1e6"
                ),
            },
            "events": event_metadata,
            "selected_assets": selected_assets,
            "simulation_config": simulation_config,
            "code_fingerprints": {
                "dynamic_vla_sha256": sha256_code_tree(args.dynamic_vla_root),
                "event_generator_sha256": sha256_code_tree(args.event_code_root),
            },
            "software": {
                "python": platform.python_version(),
                "isaac_sim": package_version("isaacsim"),
                "isaac_lab": package_version("isaaclab"),
                "torch": package_version("torch"),
                "numpy": package_version("numpy"),
                "h5py": package_version("h5py"),
                "pyarrow": package_version("pyarrow"),
                "dv_processing": package_version("dv-processing"),
                "imageio": package_version("imageio"),
            },
            "video": {
                "codec": "h264",
                "fps": FPS,
                "pixel_format": "yuv420p",
                "resolution_wh": [WIDTH, HEIGHT],
            },
            "file_sha256": files,
            "reproduction_command": (
                f"bash dynamic-vla/scripts/generate_edv_samples.sh "
                f"{args.row} 1 {args.device}"
            ),
            "determinism_note": (
                "Scene/configuration are reproducible; GPU rendering may not be bitwise "
                "identical across Isaac, driver, or hardware versions."
            ),
        }
        reproduction_path = temp_sample / "reproduction.json"
        with reproduction_path.open("w", encoding="utf-8") as stream:
            json.dump(reproduction, stream, ensure_ascii=False, indent=2)

        temp_sample.replace(final_sample)
        write_dataset_info(root)
        total_bytes = sum(
            path.stat().st_size for path in final_sample.rglob("*") if path.is_file()
        )
        print(
            f"[EDV] {sample_name} ready: frames={frame_count} "
            f"events={sum(v['event_count'] for v in event_metadata.values())} "
            f"size={total_bytes / 1024**2:.2f} MiB",
            flush=True,
        )
    except Exception:
        if temp_sample.exists():
            shutil.rmtree(temp_sample)
        raise


if __name__ == "__main__":
    main()
