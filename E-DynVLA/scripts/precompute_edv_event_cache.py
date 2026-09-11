"""Build versioned wrist-event separation caches before model training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))

from policies.edynvla.data import EventWindowConfig  # noqa: E402
from policies.edynvla.edv_support import EDVSupportDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--v2e-root", type=Path, default=REPO_ROOT / "V2E-VLA")
    parser.add_argument("--sensor", default="wrist_cam")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    dataset = EDVSupportDataset(
        args.dataset_root,
        split=None,
        event_config=EventWindowConfig(
            sensor=args.sensor,
            event_code_root=str(args.v2e_root),
        ),
        cameras=(args.sensor,),
        event_sensor=args.sensor,
        event_code_root=args.v2e_root,
        event_cache_root=args.cache_root,
        exclude_failures=not args.include_failures,
    )
    selected = [
        sample
        for sample in dataset.samples
        if int(sample["sample_index"]) >= args.start_index
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    started = time.perf_counter()
    for position, sample in enumerate(selected, start=1):
        tick = time.perf_counter()
        path, separated = dataset._ensure_event_h5(sample)
        if not separated:
            raise RuntimeError(f"Could not separate events for {sample['relative_path']}")
        print(
            f"[{position}/{len(selected)}] {sample['relative_path']} -> {path} "
            f"({time.perf_counter() - tick:.1f}s)",
            flush=True,
        )
    print(
        f"Prepared {len(selected)} caches in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
