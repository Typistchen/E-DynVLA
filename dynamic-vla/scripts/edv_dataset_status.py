#!/usr/bin/env python3
"""Print compact status for a split EDV dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--target-gib", type=float, default=500.0)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    result = {"target_gib": args.target_gib, "splits": {}}
    total_bytes = 0
    for split in ("success", "failure"):
        samples = sorted((root / split).glob("sample_*"))
        size_bytes = directory_size(root / split)
        total_bytes += size_bytes
        result["splits"][split] = {
            "samples": len(samples),
            "size_gib": size_bytes / 1024**3,
            "last_sample": samples[-1].name if samples else None,
        }
    result["total_samples"] = sum(
        split["samples"] for split in result["splits"].values()
    )
    result["total_gib"] = total_bytes / 1024**3
    result["progress_percent"] = min(100.0, result["total_gib"] / args.target_gib * 100)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
