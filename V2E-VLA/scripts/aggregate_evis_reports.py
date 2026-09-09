"""Aggregate EVIS episode reports with the episode as the independent unit.

Each camera is first averaged within an episode (ignoring unavailable values),
then paired v3-v2 differences are summarized across episodes.  This prevents a
missing camera ROI or a long/high-event-rate episode from silently becoming an
extra statistical replicate.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # The descriptive report still works without SciPy.
    wilcoxon = None


METRICS = {
    "events_per_camera": ("event", "events", "neutral"),
    "polarity_imbalance": ("event", "polarity_imbalance", "lower"),
    "count_10ms_cv": ("event", "count_10ms_cv", "lower"),
    "phase_10ms_imbalance": ("event", "phase_10ms_imbalance", "lower"),
    "harmonic_power_ratio": ("event", "harmonic_power_ratio", "lower"),
    "keyframe_25hz_db": ("event", "keyframe_25hz_db_over_local", "lower"),
    "object_boundary_coverage": ("roi", "object_boundary_coverage", "higher"),
    "object_boundary_density": (
        "roi",
        "object_boundary_events_per_pixel_interval",
        "higher",
    ),
    "halo_to_boundary_ratio": ("roi", "halo_to_boundary_density_ratio", "lower"),
    "static_event_rate": ("roi", "static_event_rate_per_mpix_s", "lower"),
    "voxel_f1_1px_1ms": ("voxel", "symmetric_f1", "neutral"),
    "voxel_precision_1px_1ms": ("voxel", "v3_precision_vs_v2", "neutral"),
    "voxel_recall_1px_1ms": ("voxel", "v3_recall_vs_v2", "neutral"),
}


def _finite_mean(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _episode_value(report, version, kind, key, camera=None):
    cameras = [camera] if camera is not None else report["cameras"]
    if kind == "event":
        return _finite_mean([report[version]["cameras"][name].get(key) for name in cameras])
    if kind == "roi":
        return _finite_mean(
            [report["roi_proxies"][version][name].get(key) for name in cameras]
        )
    if version == "v2":
        return None
    return _finite_mean(
        [report["voxel_agreement"][name]["1px_1ms"].get(key) for name in cameras]
    )


def _paired_summary(v2_values, v3_values, direction, rng, bootstrap_samples):
    pairs = [
        (float(v2), float(v3))
        for v2, v3 in zip(v2_values, v3_values)
        if v2 is not None and v3 is not None and np.isfinite(v2) and np.isfinite(v3)
    ]
    if not pairs:
        return None
    v2 = np.asarray([pair[0] for pair in pairs])
    v3 = np.asarray([pair[1] for pair in pairs])
    delta = v3 - v2
    samples = delta[rng.integers(0, delta.size, (bootstrap_samples, delta.size))].mean(axis=1)
    if direction == "lower":
        improved = int(np.count_nonzero(delta < 0))
    elif direction == "higher":
        improved = int(np.count_nonzero(delta > 0))
    else:
        improved = None
    p_value = None
    if wilcoxon is not None and np.any(delta != 0):
        # scipy cannot use its exact path when zeros are present. Dropping zeros
        # explicitly is the standard Wilcox convention and preserves an exact
        # small-sample result for the remaining paired differences.
        nonzero_delta = delta[delta != 0]
        p_value = float(
            wilcoxon(nonzero_delta, alternative="two-sided", method="exact").pvalue
        )
    return {
        "paired_episodes": int(delta.size),
        "v2_mean": float(v2.mean()),
        "v3_mean": float(v3.mean()),
        "mean_delta_v3_minus_v2": float(delta.mean()),
        "median_delta_v3_minus_v2": float(np.median(delta)),
        "relative_mean_delta": float(delta.mean() / v2.mean()) if v2.mean() else None,
        "bootstrap_95ci_mean_delta": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "direction": direction,
        "episodes_improved": improved,
        "episodes_increased": int(np.count_nonzero(delta > 0)),
        "episodes_decreased": int(np.count_nonzero(delta < 0)),
        "wilcoxon_two_sided_p": p_value,
        "paired_deltas": delta.tolist(),
    }


def _fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _build_markdown(summary):
    lines = [
        "# EVIS v2 vs v3 multi-episode paired summary",
        "",
        f"- Episodes: `{summary['episodes']}`",
        f"- State/action exact in every episode: `{summary['fairness']['state_arrays_exact_all']}`",
        f"- Segmentation exact in every episode: `{summary['fairness']['segmentation_exact_all']}`",
        "- Independent unit: one episode; cameras are averaged within each episode.",
        "- CI: paired episode bootstrap; p: paired two-sided Wilcoxon (exploratory).",
        "",
        "| Metric | Direction | v2 mean | v3 mean | Mean delta | Median delta | 95% CI | Improved | p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in summary["metrics"].items():
        if item is None:
            continue
        ci = item["bootstrap_95ci_mean_delta"]
        improved = (
            "n/a"
            if item["episodes_improved"] is None
            else f"{item['episodes_improved']}/{item['paired_episodes']}"
        )
        lines.append(
            f"| {name} | {item['direction']} | {_fmt(item['v2_mean'])} | "
            f"{_fmt(item['v3_mean'])} | {_fmt(item['mean_delta_v3_minus_v2'])} | "
            f"{_fmt(item['median_delta_v3_minus_v2'])} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {improved} | "
            f"{_fmt(item['wilcoxon_two_sided_p'], 5)} |"
        )

    lines += [
        "",
        "## Per-camera mean changes",
        "",
        "These rows are diagnostic views of the same episodes, not additional independent samples.",
        "",
        "| Metric | Camera | Mean delta | Improved | p |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for metric, camera_items in summary["per_camera"].items():
        for camera, item in camera_items.items():
            if item is None:
                continue
            improved = (
                "n/a"
                if item["episodes_improved"] is None
                else f"{item['episodes_improved']}/{item['paired_episodes']}"
            )
            lines.append(
                f"| {metric} | {camera} | {_fmt(item['mean_delta_v3_minus_v2'])} | "
                f"{improved} | {_fmt(item['wilcoxon_two_sided_p'], 5)} |"
            )
    lines += [
        "",
        "Voxel precision/recall/F1 measure similarity to v2, not accuracy against an oracle.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        parser.error(f"No reports matched: {args.input_glob}")
    reports = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as stream:
            reports.append(json.load(stream))
    rng = np.random.default_rng(args.bootstrap_seed)
    metrics = {}
    per_camera = {}
    for name, (kind, key, direction) in METRICS.items():
        v2 = [_episode_value(report, "v2", kind, key) for report in reports]
        v3 = [_episode_value(report, "v3", kind, key) for report in reports]
        if kind == "voxel":
            metrics[name] = {
                "paired_episodes": len(reports),
                "v2_mean": None,
                "v3_mean": _finite_mean(v3),
                "mean_delta_v3_minus_v2": None,
                "median_delta_v3_minus_v2": None,
                "relative_mean_delta": None,
                "bootstrap_95ci_mean_delta": [None, None],
                "direction": direction,
                "episodes_improved": None,
                "episodes_increased": None,
                "episodes_decreased": None,
                "wilcoxon_two_sided_p": None,
                "paired_deltas": None,
            }
            continue
        metrics[name] = _paired_summary(
            v2, v3, direction, rng, args.bootstrap_samples
        )
        per_camera[name] = {}
        for camera in reports[0]["cameras"]:
            camera_v2 = [
                _episode_value(report, "v2", kind, key, camera) for report in reports
            ]
            camera_v3 = [
                _episode_value(report, "v3", kind, key, camera) for report in reports
            ]
            per_camera[name][camera] = _paired_summary(
                camera_v2, camera_v3, direction, rng, args.bootstrap_samples
            )

    summary = {
        "schema_version": 1,
        "episodes": len(reports),
        "input_reports": paths,
        "fairness": {
            "state_arrays_exact_all": all(
                report["episode_consistency"]["state_arrays_exact"] for report in reports
            ),
            "segmentation_exact_all": all(
                report["episode_consistency"]["segmentation_exact"] for report in reports
            ),
        },
        "aggregation": "mean cameras within episode, then paired episodes",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "metrics": metrics,
        "per_camera": per_camera,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    with open(args.out_md, "w", encoding="utf-8") as stream:
        stream.write(_build_markdown(summary))
    print(f"[aggregate_evis_reports] JSON -> {args.out_json}")
    print(f"[aggregate_evis_reports] Markdown -> {args.out_md}")


if __name__ == "__main__":
    main()
