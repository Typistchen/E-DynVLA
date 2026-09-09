"""Compare two EVIS event-generation versions without a downstream task.

The evaluator intentionally separates metrics that can be computed from the
existing DOM episode from metrics that require a high-rate rendered oracle.
It reports event statistics, temporal/keyframe artifacts, cup-boundary support,
static/halo leakage proxies, optional confidence statistics, and cross-version
voxel agreement. It does *not* label any version as ground truth.

Example::

    python scripts/evaluate_evis_versions.py \
      --v2-events /path/v2/events/env0_ep2.h5 \
      --v2-episode /path/v2/episode.h5 \
      --v3-events /path/v3/events/env0_ep2.h5 \
      --v3-episode /path/v3/episode.h5 \
      --event-time-origin-s 0.07 \
      --out-json /tmp/evis_metrics.json --out-md /tmp/evis_metrics.md

``--event-time-origin-s`` converts Isaac's absolute simulation timestamps to
episode-relative time.  Per-version overrides are available when two captures
started at different simulation times.  If no CLI value is supplied, the
evaluator reads ``event_time_origin_s``/``time_origin_s`` from the event HDF5
root or ``/DVS`` attributes, and finally falls back to zero.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERA_ORDER = ("opst_cam", "side_cam", "wrist_cam")
OBJECT_LABEL = 2


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _json_safe(value):
    """Recursively replace non-finite numbers so output is strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _event_group(stream: h5py.File, camera: str):
    return stream[f"DVS/{camera}"]


def _camera_names(path: str) -> list[str]:
    with h5py.File(path, "r") as stream:
        present = set(stream["DVS"].keys())
    return [camera for camera in CAMERA_ORDER if camera in present] + sorted(
        present - set(CAMERA_ORDER)
    )


def _resolve_time_origin(path: str, cli_value: float | None):
    if cli_value is not None:
        return float(cli_value), "cli"
    with h5py.File(path, "r") as stream:
        for owner_name, owner in (("root", stream), ("DVS", stream.get("DVS"))):
            if owner is None:
                continue
            for key in ("event_time_origin_s", "time_origin_s"):
                if key in owner.attrs:
                    return float(owner.attrs[key]), f"hdf5:{owner_name}:{key}"
    return 0.0, "default_zero"


def _event_source_metadata(path: str):
    with h5py.File(path, "r") as stream:
        metadata = {}
        for key, value in stream.attrs.items():
            if not str(key).startswith("evis_"):
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            metadata[str(key)] = value
    return metadata


def _harmonic_metrics(times: np.ndarray, duration: float, keyframe_hz: float):
    """Return keyframe-line power from a 1 ms event-count signal.

    The ratio uses 5--250 Hz non-DC power as its denominator and integrates a
    +/-0.75 Hz band around the keyframe frequency and its harmonics. A Hann
    window reduces leakage from the finite episode boundary.
    """
    bin_s = 1e-3
    edges = np.arange(0.0, duration + bin_s * 1.01, bin_s)
    counts = np.histogram(times, bins=edges)[0].astype(np.float64)
    signal = counts - counts.mean()
    if signal.size < 4 or not np.any(signal):
        return {"harmonic_power_ratio": 0.0, "keyframe_25hz_db_over_local": 0.0}
    spectrum = np.fft.rfft(signal * np.hanning(signal.size))
    power = np.square(np.abs(spectrum))
    freq = np.fft.rfftfreq(signal.size, d=bin_s)
    denominator = (freq >= 5.0) & (freq <= 250.0)
    lines = np.zeros_like(denominator)
    harmonic = keyframe_hz
    while harmonic <= 250.0:
        lines |= np.abs(freq - harmonic) <= 0.75
        harmonic += keyframe_hz
    harmonic_ratio = _safe_ratio(power[lines & denominator].sum(), power[denominator].sum())

    line25 = np.abs(freq - keyframe_hz) <= 0.75
    local = (freq >= keyframe_hz - 5.0) & (freq <= keyframe_hz + 5.0)
    local &= np.abs(freq - keyframe_hz) >= 1.5
    line_mean = float(power[line25].mean()) if np.any(line25) else 0.0
    local_median = float(np.median(power[local])) if np.any(local) else 0.0
    line_db = 10.0 * math.log10((line_mean + 1e-12) / (local_median + 1e-12))
    return {
        "harmonic_power_ratio": float(harmonic_ratio),
        "keyframe_25hz_db_over_local": float(line_db),
    }


def summarize_events(
    path: str,
    cameras: list[str],
    duration: float,
    keyframe_hz: float,
    time_origin_s: float,
):
    result = {
        "path": path,
        "file_bytes": os.path.getsize(path),
        "total_events": 0,
        "cameras": {},
    }
    with h5py.File(path, "r") as stream:
        for camera in cameras:
            group = _event_group(stream, camera)
            raw_times = group["t"][:]
            times = raw_times - time_origin_s
            polarity = group["p"][:]
            n_events = int(times.size)
            n_on = int(np.count_nonzero(polarity > 0))
            n_off = int(np.count_nonzero(polarity < 0))
            result["total_events"] += n_events

            edges = np.arange(0.0, duration + 0.010001, 0.01)
            counts = np.histogram(times, bins=edges)[0].astype(np.float64)
            phase_means = [float(counts[index::4].mean()) for index in range(4)]
            summary = {
                "events": n_events,
                "event_rate_per_s": _safe_ratio(n_events, duration),
                "on_events": n_on,
                "off_events": n_off,
                "polarity_imbalance": _safe_ratio(abs(n_on - n_off), n_events),
                "timestamps_monotonic": bool(np.all(times[1:] >= times[:-1])),
                "timestamp_origin_s": time_origin_s,
                "timestamp_min_s": float(times[0]) if n_events else None,
                "timestamp_max_s": float(times[-1]) if n_events else None,
                "timestamp_min_raw_s": float(raw_times[0]) if n_events else None,
                "timestamp_max_raw_s": float(raw_times[-1]) if n_events else None,
                "events_in_analysis_window": int(
                    np.count_nonzero((times >= 0.0) & (times <= duration))
                ),
                "count_10ms_cv": _safe_ratio(float(counts.std()), float(counts.mean())),
                "phase_10ms_means": phase_means,
                "phase_10ms_imbalance": _safe_ratio(max(phase_means), min(phase_means)),
                **_harmonic_metrics(times, duration, keyframe_hz),
            }
            if "q" in group:
                quality = group["q"][:].astype(np.float32)
                quantiles = np.quantile(quality, [0.01, 0.10, 0.50, 0.90, 0.99])
                summary["confidence"] = {
                    "mean": float(quality.mean()),
                    "min": float(quality.min()),
                    "q01": float(quantiles[0]),
                    "q10": float(quantiles[1]),
                    "q50": float(quantiles[2]),
                    "q90": float(quantiles[3]),
                    "q99": float(quantiles[4]),
                    "fraction_at_floor_0_5": float(np.mean(quality <= 0.5001)),
                    "fraction_ge_0_9": float(np.mean(quality >= 0.9)),
                }
            result["cameras"][camera] = summary
    result["bytes_per_event"] = _safe_ratio(result["file_bytes"], result["total_events"])
    return result


def _kernel(radius: int):
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _dilate(mask: np.ndarray, radius: int):
    if radius <= 0:
        return mask
    return cv2.dilate(mask.astype(np.uint8), _kernel(radius)) > 0


def _semantic_edges(seg: np.ndarray):
    edge = np.zeros_like(seg, dtype=bool)
    edge[1:] |= seg[1:] != seg[:-1]
    edge[:-1] |= seg[:-1] != seg[1:]
    edge[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    edge[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    return edge


def _log_luminance(rgb: np.ndarray):
    rgb = rgb.astype(np.float32)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.log(lum + 1.0)


def roi_proxy_metrics(
    event_path: str,
    episode_path: str,
    cameras: list[str],
    frame_dt: float,
    time_origin_s: float,
):
    """Compute segmentation/RGB-supported proxies, not oracle accuracy metrics."""
    output = {}
    with h5py.File(event_path, "r") as events, h5py.File(episode_path, "r") as episode:
        for camera in cameras:
            group = _event_group(events, camera)
            xs = group["x"][:].astype(np.int32)
            ys = group["y"][:].astype(np.int32)
            times = group["t"][:] - time_origin_s
            quality = group["q"][:].astype(np.float32) if "q" in group else None
            seg_ds = episode[f"{camera}_seg"]
            rgb_ds = episode[f"{camera}_rgb"]
            n_frames, height, width, _ = seg_ds.shape

            totals = {
                "active_object_intervals": 0,
                "object_boundary_pixels": 0,
                "object_boundary_covered_pixels": 0,
                "object_boundary_events": 0,
                "object_boundary_weighted_events": 0.0,
                "object_roi_events": 0,
                "object_roi_weighted_events": 0.0,
                "halo_pixels": 0,
                "halo_events": 0,
                "halo_weighted_events": 0.0,
                "static_pixels": 0,
                "static_events": 0,
                "static_weighted_events": 0.0,
                "all_interval_events": 0,
            }
            for index in range(n_frames - 1):
                t0, t1 = index * frame_dt, (index + 1) * frame_dt
                lo = int(np.searchsorted(times, t0, side="left"))
                hi = int(np.searchsorted(times, t1, side="left"))
                if hi <= lo:
                    event_count = np.zeros((height, width), dtype=np.int32)
                    weighted_count = np.zeros((height, width), dtype=np.float32)
                else:
                    flat = ys[lo:hi] * width + xs[lo:hi]
                    valid = (xs[lo:hi] < width) & (ys[lo:hi] < height)
                    flat = flat[valid]
                    event_count = np.bincount(flat, minlength=height * width).reshape(height, width)
                    weights = quality[lo:hi][valid] if quality is not None else np.ones(flat.size)
                    weighted_count = np.bincount(
                        flat, weights=weights, minlength=height * width
                    ).reshape(height, width)
                totals["all_interval_events"] += int(event_count.sum())

                seg0 = seg_ds[index, ..., 0]
                seg1 = seg_ds[index + 1, ..., 0]
                obj0, obj1 = seg0 == OBJECT_LABEL, seg1 == OBJECT_LABEL
                object_motion = int(np.count_nonzero(obj0 ^ obj1))
                if (obj0.any() or obj1.any()) and object_motion >= 10:
                    totals["active_object_intervals"] += 1
                    edge = _semantic_edges(obj0.astype(np.uint8)) | _semantic_edges(
                        obj1.astype(np.uint8)
                    )
                    boundary = _dilate(edge, 2)
                    swept = obj0 | obj1
                    object_roi = _dilate(swept, 2)
                    # Background-only annulus 3--8 px outside the swept cup.
                    halo = _dilate(swept, 8) & ~_dilate(swept, 2)
                    halo &= (seg0 == 0) & (seg1 == 0)

                    totals["object_boundary_pixels"] += int(boundary.sum())
                    totals["object_boundary_covered_pixels"] += int(
                        np.count_nonzero(boundary & (event_count > 0))
                    )
                    totals["object_boundary_events"] += int(event_count[boundary].sum())
                    totals["object_boundary_weighted_events"] += float(
                        weighted_count[boundary].sum()
                    )
                    totals["object_roi_events"] += int(event_count[object_roi].sum())
                    totals["object_roi_weighted_events"] += float(
                        weighted_count[object_roi].sum()
                    )
                    totals["halo_pixels"] += int(halo.sum())
                    totals["halo_events"] += int(event_count[halo].sum())
                    totals["halo_weighted_events"] += float(weighted_count[halo].sum())

                rgb0 = rgb_ds[index]
                rgb1 = rgb_ds[index + 1]
                log_delta = np.abs(_log_luminance(rgb1) - _log_luminance(rgb0))
                semantic_edge = _dilate(_semantic_edges(seg0) | _semantic_edges(seg1), 3)
                # Stored RGB is LDR while events use HDR, so this is explicitly a
                # leakage proxy: unchanged class, away from edges, <2% log change.
                static = (seg0 == seg1) & ~semantic_edge & (log_delta < 0.02)
                totals["static_pixels"] += int(static.sum())
                totals["static_events"] += int(event_count[static].sum())
                totals["static_weighted_events"] += float(weighted_count[static].sum())

            boundary_density = _safe_ratio(
                totals["object_boundary_events"], totals["object_boundary_pixels"]
            )
            halo_density = _safe_ratio(totals["halo_events"], totals["halo_pixels"])
            static_exposure = totals["static_pixels"] * frame_dt
            output[camera] = {
                **totals,
                "object_boundary_coverage": _safe_ratio(
                    totals["object_boundary_covered_pixels"], totals["object_boundary_pixels"]
                ),
                "object_boundary_events_per_pixel_interval": boundary_density,
                "object_boundary_mean_q": _safe_ratio(
                    totals["object_boundary_weighted_events"], totals["object_boundary_events"]
                ),
                "object_roi_mean_q": _safe_ratio(
                    totals["object_roi_weighted_events"], totals["object_roi_events"]
                ),
                "halo_events_per_pixel_interval": halo_density,
                "halo_to_boundary_density_ratio": _safe_ratio(halo_density, boundary_density),
                "halo_mean_q": _safe_ratio(
                    totals["halo_weighted_events"], totals["halo_events"]
                ),
                "static_event_rate_per_mpix_s": _safe_ratio(
                    totals["static_events"] * 1e6, static_exposure
                ),
                "static_weighted_event_rate_per_mpix_s": _safe_ratio(
                    totals["static_weighted_events"] * 1e6, static_exposure
                ),
                "static_mean_q": _safe_ratio(
                    totals["static_weighted_events"], totals["static_events"]
                ),
            }
    return output


def _occupied_voxels(
    group,
    width: int,
    height: int,
    spatial_px: int,
    temporal_ms: int,
    time_origin_s: float,
    duration_s: float,
):
    event_time = group["t"][:] - time_origin_s
    valid = (event_time >= 0.0) & (event_time <= duration_s)
    x = group["x"][:][valid].astype(np.int64) // spatial_px
    y = group["y"][:][valid].astype(np.int64) // spatial_px
    t = np.floor(event_time[valid] / (temporal_ms * 1e-3)).astype(np.int64)
    p = (group["p"][:][valid] > 0).astype(np.int64)
    nx = math.ceil(width / spatial_px)
    ny = math.ceil(height / spatial_px)
    code = (((t * ny + y) * nx + x) << 1) | p
    return np.unique(code)


def voxel_agreement(
    v2_path: str,
    v3_path: str,
    cameras: list[str],
    width: int,
    height: int,
    v2_time_origin_s: float,
    v3_time_origin_s: float,
    duration_s: float,
):
    configs = ((1, 1), (2, 2), (4, 5))
    result = {}
    with h5py.File(v2_path, "r") as v2, h5py.File(v3_path, "r") as v3:
        for camera in cameras:
            camera_result = {}
            for spatial_px, temporal_ms in configs:
                a = _occupied_voxels(
                    _event_group(v2, camera),
                    width,
                    height,
                    spatial_px,
                    temporal_ms,
                    v2_time_origin_s,
                    duration_s,
                )
                b = _occupied_voxels(
                    _event_group(v3, camera),
                    width,
                    height,
                    spatial_px,
                    temporal_ms,
                    v3_time_origin_s,
                    duration_s,
                )
                common = int(np.intersect1d(a, b, assume_unique=True).size)
                precision = _safe_ratio(common, b.size)
                recall = _safe_ratio(common, a.size)
                f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
                camera_result[f"{spatial_px}px_{temporal_ms}ms"] = {
                    "v2_occupied": int(a.size),
                    "v3_occupied": int(b.size),
                    "intersection": common,
                    "v3_precision_vs_v2": precision,
                    "v3_recall_vs_v2": recall,
                    "symmetric_f1": f1,
                }
            result[camera] = camera_result
    return result


def episode_consistency(v2_path: str, v3_path: str, cameras: list[str]):
    result = {"state_arrays_exact": True, "segmentation_exact": True, "cameras": {}}
    state_keys = ("action", "ee_pos", "object_pos", "object_quat", "object_vel", "sm_state")
    with h5py.File(v2_path, "r") as v2, h5py.File(v3_path, "r") as v3:
        result["n_frames"] = int(v2["action"].shape[0])
        for key in state_keys:
            exact = bool(np.array_equal(v2[key][:], v3[key][:]))
            result[f"{key}_exact"] = exact
            result["state_arrays_exact"] &= exact
        for camera in cameras:
            seg_exact = bool(np.array_equal(v2[f"{camera}_seg"][:], v3[f"{camera}_seg"][:]))
            result["segmentation_exact"] &= seg_exact
            rgb_a = v2[f"{camera}_rgb"][:].astype(np.int16)
            rgb_b = v3[f"{camera}_rgb"][:].astype(np.int16)
            result["cameras"][camera] = {
                "segmentation_exact": seg_exact,
                "rgb_mae_u8": float(np.abs(rgb_a - rgb_b).mean()),
                "rgb_exact_fraction": float(np.mean(rgb_a == rgb_b)),
            }
        shape = v2[f"{cameras[0]}_seg"].shape
        result["height"] = int(shape[1])
        result["width"] = int(shape[2])
    return result


def _fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def build_markdown(metrics: dict) -> str:
    lines = [
        "# EVIS v2 vs v3 source-level evaluation",
        "",
        "These are episode-level, no-downstream-task metrics. Event-F1, timestamp MAE, and",
        "log-intensity RMSE versus truth remain unavailable until a high-rate rendered oracle exists.",
        "Segmentation/state equality makes the present comparison controlled.",
        "",
        "## Fairness check",
        "",
        f"- State/action arrays exact: `{metrics['episode_consistency']['state_arrays_exact']}`",
        f"- Segmentation exact: `{metrics['episode_consistency']['segmentation_exact']}`",
        f"- v2 event time origin: `{metrics['time_alignment']['v2_origin_s']:.6f} s` "
        f"(`{metrics['time_alignment']['v2_origin_source']}`)",
        f"- v3 event time origin: `{metrics['time_alignment']['v3_origin_s']:.6f} s` "
        f"(`{metrics['time_alignment']['v3_origin_source']}`)",
        f"- v2 recorded EVIS mode: "
        f"`{metrics['source_metadata']['v2'].get('evis_mode', 'unrecorded')}`",
        f"- v3 recorded EVIS mode: "
        f"`{metrics['source_metadata']['v3'].get('evis_mode', 'unrecorded')}`",
        "",
        "## Event and temporal metrics",
        "",
        "| Camera | Version | Events | ON/OFF imbalance | 10 ms CV | Phase imbalance | Harmonic power | 25 Hz dB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for camera in metrics["cameras"]:
        for version in ("v2", "v3"):
            item = metrics[version]["cameras"][camera]
            lines.append(
                f"| {camera} | {version} | {item['events']:,} | "
                f"{_fmt(item['polarity_imbalance'])} | {_fmt(item['count_10ms_cv'])} | "
                f"{_fmt(item['phase_10ms_imbalance'])} | "
                f"{_fmt(item['harmonic_power_ratio'])} | "
                f"{_fmt(item['keyframe_25hz_db_over_local'], 2)} |"
            )
    lines += [
        "",
        "## Cup-boundary and leakage proxies",
        "",
        "Boundary/halo values use DOM segmentation; static leakage uses stored LDR RGB",
        "while the event source is HDR, so these are controlled proxies rather than oracle accuracy.",
        "",
        "| Camera | Version | Cup boundary coverage | Boundary density | Halo/boundary | Static events / MPix/s | q-weighted static |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for camera in metrics["cameras"]:
        for version in ("v2", "v3"):
            item = metrics["roi_proxies"][version][camera]
            lines.append(
                f"| {camera} | {version} | {_fmt(item['object_boundary_coverage'])} | "
                f"{_fmt(item['object_boundary_events_per_pixel_interval'])} | "
                f"{_fmt(item['halo_to_boundary_density_ratio'])} | "
                f"{_fmt(item['static_event_rate_per_mpix_s'], 1)} | "
                f"{_fmt(item['static_weighted_event_rate_per_mpix_s'], 1)} |"
            )
    lines += [
        "",
        "## Cross-version occupied-voxel agreement",
        "",
        "This measures v2/v3 similarity, not correctness against real events.",
        "",
        "| Camera | Resolution | Symmetric F1 | v3 precision vs v2 | v3 recall vs v2 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for camera in metrics["cameras"]:
        for resolution, item in metrics["voxel_agreement"][camera].items():
            lines.append(
                f"| {camera} | {resolution} | {_fmt(item['symmetric_f1'])} | "
                f"{_fmt(item['v3_precision_vs_v2'])} | {_fmt(item['v3_recall_vs_v2'])} |"
            )
    confidence = metrics["v3"]["cameras"]
    lines += [
        "",
        "## v3 confidence",
        "",
        "| Camera | Mean q | Median q | q=0.5 fraction | q>=0.9 fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for camera in metrics["cameras"]:
        q = confidence[camera].get("confidence", {})
        lines.append(
            f"| {camera} | {_fmt(q.get('mean'))} | {_fmt(q.get('q50'))} | "
            f"{_fmt(q.get('fraction_at_floor_0_5'))} | {_fmt(q.get('fraction_ge_0_9'))} |"
        )
    lines += [
        "",
        "## Metrics requiring a new high-rate oracle run",
        "",
        "- Event precision / recall / F1 against truth",
        "- Matched-event timestamp MAE",
        "- Log-intensity trajectory RMSE",
        "- q calibration (ECE, Brier score, AUROC)",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-events", required=True)
    parser.add_argument("--v2-episode", required=True)
    parser.add_argument("--v3-events", required=True)
    parser.add_argument("--v3-episode", required=True)
    parser.add_argument("--frame-dt", type=float, default=0.04)
    parser.add_argument("--keyframe-hz", type=float, default=25.0)
    parser.add_argument(
        "--event-time-origin-s",
        type=float,
        default=None,
        help="Shared absolute simulation timestamp corresponding to episode t=0.",
    )
    parser.add_argument(
        "--v2-time-origin-s",
        type=float,
        default=None,
        help="Override the shared event time origin for v2.",
    )
    parser.add_argument(
        "--v3-time-origin-s",
        type=float,
        default=None,
        help="Override the shared event time origin for v3.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    cameras = [
        camera
        for camera in _camera_names(args.v2_events)
        if camera in _camera_names(args.v3_events)
    ]
    consistency = episode_consistency(args.v2_episode, args.v3_episode, cameras)
    duration = consistency["n_frames"] * args.frame_dt
    v2_origin_arg = (
        args.v2_time_origin_s
        if args.v2_time_origin_s is not None
        else args.event_time_origin_s
    )
    v3_origin_arg = (
        args.v3_time_origin_s
        if args.v3_time_origin_s is not None
        else args.event_time_origin_s
    )
    v2_origin_s, v2_origin_source = _resolve_time_origin(args.v2_events, v2_origin_arg)
    v3_origin_s, v3_origin_source = _resolve_time_origin(args.v3_events, v3_origin_arg)
    metrics = {
        "schema_version": 2,
        "cameras": cameras,
        "frame_dt_s": args.frame_dt,
        "keyframe_hz": args.keyframe_hz,
        "episode_duration_s": duration,
        "time_alignment": {
            "v2_origin_s": v2_origin_s,
            "v3_origin_s": v3_origin_s,
            "v2_origin_source": v2_origin_source,
            "v3_origin_source": v3_origin_source,
        },
        "source_metadata": {
            "v2": _event_source_metadata(args.v2_events),
            "v3": _event_source_metadata(args.v3_events),
        },
        "episode_consistency": consistency,
        "v2": summarize_events(
            args.v2_events, cameras, duration, args.keyframe_hz, v2_origin_s
        ),
        "v3": summarize_events(
            args.v3_events, cameras, duration, args.keyframe_hz, v3_origin_s
        ),
        "roi_proxies": {
            "v2": roi_proxy_metrics(
                args.v2_events, args.v2_episode, cameras, args.frame_dt, v2_origin_s
            ),
            "v3": roi_proxy_metrics(
                args.v3_events, args.v3_episode, cameras, args.frame_dt, v3_origin_s
            ),
        },
        "voxel_agreement": voxel_agreement(
            args.v2_events,
            args.v3_events,
            cameras,
            consistency["width"],
            consistency["height"],
            v2_origin_s,
            v3_origin_s,
            duration,
        ),
        "unavailable_without_high_rate_oracle": [
            "event_precision_recall_f1",
            "matched_timestamp_mae",
            "log_intensity_rmse",
            "confidence_ece_brier_auroc",
        ],
    }

    metrics = _json_safe(metrics)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False, allow_nan=False)
    with open(args.out_md, "w", encoding="utf-8") as stream:
        stream.write(build_markdown(metrics))
    print(f"[evaluate_evis_versions] JSON -> {args.out_json}")
    print(f"[evaluate_evis_versions] Markdown -> {args.out_md}")


if __name__ == "__main__":
    main()
