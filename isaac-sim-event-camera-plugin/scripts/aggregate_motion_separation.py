#!/usr/bin/env python3
"""Aggregate ten-demo lighting-suppressed motion-separation results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import stats


LABELS = {
    "background": (0, 3),
    "robot": (1,),
    "target": (2,),
}


def _demo_number(path: Path) -> int:
    match = re.search(r"demo(\d+)", str(path))
    if not match:
        raise ValueError(f"cannot infer demo number from {path}")
    return int(match.group(1))


def _gate_evaluation(derived_path: Path, threshold: float) -> dict[str, float]:
    totals = {name: 0 for name in LABELS}
    candidates = {name: 0 for name in LABELS}
    retained = {name: 0 for name in LABELS}
    with h5py.File(derived_path, "r") as derived:
        episode_path = Path(derived.attrs["source_episode"])
        origin = float(derived.attrs["event_time_origin_s"])
        camera = next(iter(derived["DVS"].keys()))
        group = derived[f"DVS/{camera}"]
        with h5py.File(episode_path, "r") as episode:
            segmentation = episode[f"{camera}_seg"][:]
            n_frames = segmentation.shape[0]
            length = group["x"].shape[0]
            for lo in range(0, length, 2_000_000):
                hi = min(length, lo + 2_000_000)
                x = group["x"][lo:hi]
                y = group["y"][lo:hi]
                timestamp = group["t"][lo:hi]
                q_motion = group["q_motion"][lo:hi]
                q_dynamic = group["q_dynamic"][lo:hi]
                frame = np.clip(
                    np.floor((timestamp - origin) * 25.0).astype(np.int64),
                    0,
                    n_frames - 1,
                )
                inside = (x < segmentation.shape[2]) & (y < segmentation.shape[1])
                indices = np.flatnonzero(inside)
                labels = segmentation[
                    frame[indices], y[indices], x[indices], 0
                ]
                before = q_motion[indices] >= threshold
                after = q_dynamic[indices] >= threshold
                for name, ids in LABELS.items():
                    selected = np.isin(labels, ids)
                    totals[name] += int(np.count_nonzero(selected))
                    candidates[name] += int(np.count_nonzero(before & selected))
                    retained[name] += int(np.count_nonzero(after & selected))
    result = {}
    for name in LABELS:
        total = max(totals[name], 1)
        candidate = max(candidates[name], 1)
        result[f"{name}_motion_candidate_retention"] = candidates[name] / total
        result[f"{name}_dynamic_retention"] = retained[name] / total
        result[f"{name}_removed_from_motion"] = (
            candidates[name] - retained[name]
        ) / candidate
    return result


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if array.size > 1:
        half = float(stats.t.ppf(0.975, array.size - 1) * std / np.sqrt(array.size))
    else:
        half = 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--result-dir", default="motion_separation_v2")
    parser.add_argument("--out-prefix", default="motion_separation_v2_10demo")
    args = parser.parse_args()

    metric_paths = sorted(
        args.root.glob(
            f"demo*/{args.result_dir}/wrist_cam_motion_separation_metrics.json"
        ),
        key=_demo_number,
    )
    if len(metric_paths) != 10:
        raise SystemExit(f"expected 10 metric files, found {len(metric_paths)}")

    rows = []
    for metric_path in metric_paths:
        metrics = json.loads(metric_path.read_text())
        derived_path = Path(metrics["derived_h5"])
        gate = _gate_evaluation(
            derived_path, float(metrics["event_confidence_threshold"])
        )
        rows.append(
            {
                "demo": _demo_number(metric_path),
                **metrics,
                "gate_evaluation": gate,
            }
        )

    metric_keys = [
        "dynamic_event_fraction",
        "static_event_fraction",
        "illumination_event_fraction",
        "illumination_rejected_motion_fraction",
        "eval_background_dynamic_retention",
        "eval_target_dynamic_retention",
        "eval_robot_dynamic_retention",
        "eval_moving_target_pixel_recall",
        "relative_flow_threshold",
    ]
    summary = {
        key: _summary([float(row[key]) for row in rows])
        for key in metric_keys
    }
    gate_keys = [
        f"{name}_{suffix}"
        for name in LABELS
        for suffix in (
            "motion_candidate_retention",
            "dynamic_retention",
            "removed_from_motion",
        )
    ]
    gate_summary = {
        key: _summary([float(row["gate_evaluation"][key]) for row in rows])
        for key in gate_keys
    }
    background_delta = [
        row["gate_evaluation"]["background_dynamic_retention"]
        - row["gate_evaluation"]["background_motion_candidate_retention"]
        for row in rows
    ]
    target_delta = [
        row["gate_evaluation"]["target_dynamic_retention"]
        - row["gate_evaluation"]["target_motion_candidate_retention"]
        for row in rows
    ]
    significance = {
        "background_gate_delta": {
            **_summary(background_delta),
            "wilcoxon_p": float(stats.wilcoxon(background_delta).pvalue),
        },
        "target_gate_delta": {
            **_summary(target_delta),
            "wilcoxon_p": float(stats.wilcoxon(target_delta).pvalue),
        },
    }
    report = {
        "schema_version": 1,
        "algorithm": "lighting_suppressed_ego_motion_v2",
        "n_demos": len(rows),
        "algorithm_uses_semantics": False,
        "algorithm_uses_object_velocity": False,
        "evaluation_uses_semantics": True,
        "per_demo": rows,
        "summary": summary,
        "gate_summary": gate_summary,
        "paired_gate_significance": significance,
    }
    json_path = args.root / f"{args.out_prefix}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    lines = [
        "# Lighting-suppressed motion separation: 10-demo report",
        "",
        "The algorithm uses no semantic labels or object velocity. Simulator",
        "semantics and object velocity are loaded only after inference for evaluation.",
        "Isaac motion vectors remain an observed-flow upper bound in this pilot.",
        "",
        "## Per-demo results",
        "",
        "| Demo | Dynamic events | Background retained | Target retained | "
        "Illumination events | Motion rejected by light gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['demo']} | {pct(row['dynamic_event_fraction'])} | "
            f"{pct(row['eval_background_dynamic_retention'])} | "
            f"{pct(row['eval_target_dynamic_retention'])} | "
            f"{pct(row['illumination_event_fraction'])} | "
            f"{pct(row['illumination_rejected_motion_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Ten-demo mean",
            "",
            "| Metric | Mean ± std | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, label in (
        ("eval_target_dynamic_retention", "Target-event retention"),
        ("eval_background_dynamic_retention", "Background dynamic leakage"),
        ("eval_moving_target_pixel_recall", "Moving-target pixel recall"),
        ("dynamic_event_fraction", "Dynamic-event fraction"),
        ("static_event_fraction", "Static-event fraction"),
        ("illumination_event_fraction", "Illumination-event fraction"),
        ("illumination_rejected_motion_fraction", "Motion candidates rejected"),
    ):
        item = summary[key]
        lines.append(
            f"| {label} | {pct(item['mean'])} ± {pct(item['std'])} | "
            f"[{pct(item['ci95_low'])}, {pct(item['ci95_high'])}] |"
        )
    bg_gate = gate_summary["background_removed_from_motion"]
    target_gate = gate_summary["target_removed_from_motion"]
    lines.extend(
        [
            "",
            "## Lighting-gate selectivity",
            "",
            f"- Removed from background motion candidates: **{pct(bg_gate['mean'])}**.",
            f"- Removed from target motion candidates: **{pct(target_gate['mean'])}**.",
            f"- Background retention delta after gate: "
            f"**{pct(significance['background_gate_delta']['mean'])}**, "
            f"Wilcoxon p={significance['background_gate_delta']['wilcoxon_p']:.4f}.",
            f"- Target retention delta after gate: "
            f"**{pct(significance['target_gate_delta']['mean'])}**, "
            f"Wilcoxon p={significance['target_gate_delta']['wilcoxon_p']:.4f}.",
            "",
            "## Interpretation limits",
            "",
            "- There is no per-event illumination ground truth, so the illumination",
            "  channel is a geometry/photometry proxy, not measured precision/recall.",
            "- The simulator motion-vector input is privileged. Real deployment must",
            "  replace it with estimated RGB/event flow or RGB-D scene flow.",
            "- Downstream VLA success is not measured in this report.",
            "",
        ]
    )
    md_path = args.root / f"{args.out_prefix}.md"
    md_path.write_text("\n".join(lines))
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
