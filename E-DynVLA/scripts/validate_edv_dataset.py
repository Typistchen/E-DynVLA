#!/usr/bin/env python3
"""Independently validate an EDV sample dataset and emit a JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import dv_processing as dv
import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = ("wrist_cam", "opst_cam", "side_cam")
EXPECTED_SCHEMA = pa.schema(
    [
        ("action", pa.list_(pa.float32())),
        ("observation.state", pa.list_(pa.float32())),
        ("observation.environment_state", pa.list_(pa.float32())),
        ("timestamp", pa.float64()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_aedat(path: Path, expected_count: int) -> dict:
    reader = dv.io.MonoCameraRecording(str(path))
    resolution = [int(v) for v in reader.getEventResolution()]
    count = 0
    lowest = None
    highest = None
    previous_highest = None
    monotonic = True
    while reader.isRunning():
        events = reader.getNextEventBatch()
        if events is None or events.isEmpty():
            continue
        batch_low = int(events.getLowestTime())
        batch_high = int(events.getHighestTime())
        if previous_highest is not None and batch_low < previous_highest:
            monotonic = False
        previous_highest = batch_high
        lowest = batch_low if lowest is None else min(lowest, batch_low)
        highest = batch_high if highest is None else max(highest, batch_high)
        count += int(events.size())
    if count != expected_count or resolution != [480, 360] or not monotonic:
        raise RuntimeError(
            f"AEDAT4 validation failed for {path}: count={count}/{expected_count}, "
            f"resolution={resolution}, monotonic={monotonic}"
        )
    return {
        "event_count": count,
        "resolution_wh": resolution,
        "timestamp_min_us": lowest,
        "timestamp_max_us": highest,
        "monotonic": monotonic,
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    if info["total_samples"] != args.expected_samples:
        raise RuntimeError(
            f"dataset_info total_samples={info['total_samples']}, "
            f"expected={args.expected_samples}"
        )

    report = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "samples": [],
    }
    samples = sorted(
        [*root.glob("sample_*"), *root.glob("success/sample_*"), *root.glob("failure/sample_*")],
        key=lambda path: int(path.name.removeprefix("sample_")),
    )
    if len(samples) != args.expected_samples:
        raise RuntimeError(
            f"Found {len(samples)} sample directories, expected {args.expected_samples}"
        )
    for sample in samples:
        sample_index = int(sample.name.removeprefix("sample_"))
        reproduction = json.loads(
            (sample / "reproduction.json").read_text(encoding="utf-8")
        )
        frame_count = int(reproduction["frame_count"])
        parquet_path = sample / "data" / f"episode_{sample_index:06d}.parquet"
        table = pq.read_table(parquet_path)
        if table.schema != EXPECTED_SCHEMA or table.num_rows != frame_count:
            raise RuntimeError(f"Parquet validation failed for {sample}")
        timestamps = table["timestamp"].to_numpy()
        if not np.allclose(timestamps, np.arange(frame_count) / 25, atol=1e-6):
            raise RuntimeError(f"Parquet timestamps failed for {sample}")

        videos = {}
        for camera in CAMERAS:
            video_path = sample / "rgb" / f"{camera}.mp4"
            decoded_count = 0
            for frame in iio.imiter(video_path):
                if frame.shape != (360, 480, 3):
                    raise RuntimeError(f"Bad video frame shape in {video_path}: {frame.shape}")
                decoded_count += 1
            if decoded_count != frame_count:
                raise RuntimeError(
                    f"Bad frame count in {video_path}: {decoded_count}/{frame_count}"
                )
            videos[camera] = {
                "frame_count": decoded_count,
                "size_bytes": video_path.stat().st_size,
            }

        events = {}
        for camera in CAMERAS:
            events[camera] = validate_aedat(
                sample / "events" / f"{camera}.aedat4",
                int(reproduction["events"][camera]["event_count"]),
            )

        for relative, expected_hash in reproduction["file_sha256"].items():
            actual_hash = sha256_file(sample / relative)
            if actual_hash != expected_hash:
                raise RuntimeError(f"SHA256 mismatch: {sample / relative}")

        size_bytes = sum(path.stat().st_size for path in sample.rglob("*") if path.is_file())
        report["samples"].append(
            {
                "sample_index": sample_index,
                "frame_count": frame_count,
                "source_episode_index": (
                    reproduction["source_csv"]["source_episode_index"]
                    if reproduction.get("source_csv") is not None
                    else None
                ),
                "simulation_seed": reproduction["simulation_seed"],
                "outcome": reproduction.get("outcome"),
                "videos": videos,
                "events": events,
                "size_bytes": size_bytes,
                "sha256_valid": True,
            }
        )

    report["total_samples"] = len(report["samples"])
    report["total_frames"] = sum(s["frame_count"] for s in report["samples"])
    report["total_events"] = sum(
        e["event_count"] for s in report["samples"] for e in s["events"].values()
    )
    report["total_size_bytes"] = sum(s["size_bytes"] for s in report["samples"])
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[PASS] samples={report['total_samples']} frames={report['total_frames']} "
        f"events={report['total_events']} "
        f"size={report['total_size_bytes'] / 1024**2:.2f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
