#!/usr/bin/env python3
"""Validate one EVIS capture before it enters a benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED = {
    "v2_balanced": {
        "evis_mode": "v2_balanced",
        "evis_adaptive_warp": False,
        "evis_hybrid_gate_gain": 0.0,
        "evis_hybrid_support_radius": 2,
    },
    "v3_adaptive": {
        "evis_mode": "v3_adaptive",
        "evis_adaptive_warp": True,
        "evis_hybrid_gate_gain": 0.0,
        "evis_hybrid_support_radius": 2,
    },
    "v4_hybrid": {
        "evis_mode": "v4_hybrid",
        "evis_adaptive_warp": True,
        "evis_hybrid_gate_gain": 0.25,
        "evis_hybrid_support_radius": 2,
    },
}


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--expected-mode", choices=EXPECTED, required=True)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    expected = EXPECTED[args.expected_mode]
    with h5py.File(args.capture, "r") as stream:
        attrs = {str(key): _plain(value) for key, value in stream.attrs.items()}
        if "DVS" not in stream:
            raise SystemExit(f"missing /DVS in {args.capture}")
        cameras = sorted(stream["DVS"].keys())
        counts = {}
        confidence = {}
        for camera in cameras:
            group = stream[f"DVS/{camera}"]
            required = {"x", "y", "t", "p", "q"}
            missing = sorted(required - set(group.keys()))
            if missing:
                raise SystemExit(f"{camera}: missing datasets {missing}")
            lengths = {name: int(group[name].shape[0]) for name in required}
            if len(set(lengths.values())) != 1 or next(iter(lengths.values())) == 0:
                raise SystemExit(f"{camera}: invalid dataset lengths {lengths}")
            timestamps = group["t"]
            if timestamps.shape[0] > 1 and float(timestamps[-1]) < float(timestamps[0]):
                raise SystemExit(f"{camera}: timestamps are not monotonic at endpoints")
            counts[camera] = lengths["t"]
            confidence[camera] = {
                "min": float(np.min(group["q"])),
                "max": float(np.max(group["q"])),
            }

    errors = []
    for key, wanted in expected.items():
        actual = attrs.get(key)
        if isinstance(wanted, float):
            matches = actual is not None and np.isclose(float(actual), wanted)
        else:
            matches = actual == wanted
        if not matches:
            errors.append(f"{key}: expected {wanted!r}, got {actual!r}")
    if "event_time_origin_s" not in attrs:
        errors.append("missing event_time_origin_s")
    if errors:
        raise SystemExit("capture metadata mismatch:\n  " + "\n  ".join(errors))

    report = {
        "capture": str(args.capture.resolve()),
        "expected_mode": args.expected_mode,
        "metadata": attrs,
        "cameras": cameras,
        "event_counts": counts,
        "confidence_range": confidence,
        "total_events": int(sum(counts.values())),
        "valid": True,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
