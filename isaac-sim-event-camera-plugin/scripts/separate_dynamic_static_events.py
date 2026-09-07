#!/usr/bin/env python3
"""Separate static, dynamic, and illumination/unknown wrist-camera events.

This is the lighting-suppressed successor to extract_ego_dynamic_events.py.
Semantic segmentation and object velocity are optional evaluation labels only;
they never participate in calibration, threshold estimation, or classification.

The simulator pilot uses Isaac screen-space motion vectors as the observed-flow
upper bound. The remaining signals have real-world counterparts: synchronized
RGB, metric depth, camera pose, and intrinsics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from dvs_gen.io import H264Writer
from extract_ego_dynamic_events import (
    _event_image,
    _flow_image,
    _heatmap,
    _label,
    _rotation_wxyz,
    _sigmoid,
    _write_dataset,
    depth_edge_mask,
)


def ego_geometry(depth, pose0, pose1, intrinsics):
    """Return static-world forward flow, predicted next depth, and validity."""
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    z = depth
    points_c0 = np.stack(
        [(xx - cx) * z / fx, (yy - cy) * z / fy, z], axis=-1
    )
    r0 = _rotation_wxyz(pose0[3:])
    r1 = _rotation_wxyz(pose1[3:])
    points_w = points_c0 @ r0.T + pose0[:3]
    points_c1 = (points_w - pose1[:3]) @ r1
    predicted_depth = points_c1[..., 2].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        u1 = fx * points_c1[..., 0] / predicted_depth + cx
        v1 = fy * points_c1[..., 1] / predicted_depth + cy
    flow = np.stack([u1 - xx, v1 - yy], axis=-1).astype(np.float32)
    valid = (
        np.isfinite(depth)
        & np.isfinite(flow).all(axis=-1)
        & np.isfinite(predicted_depth)
        & (depth > 0.02)
        & (depth < 100.0)
        & (predicted_depth > 0.02)
        & (u1 >= -1e-4)
        & (u1 <= width - 1 + 1e-4)
        & (v1 >= -1e-4)
        & (v1 <= height - 1 + 1e-4)
    )
    flow[~valid] = 0.0
    predicted_depth[~valid] = 0.0
    return flow, predicted_depth, valid


def _sample_forward(image, flow):
    """Sample a next-frame image at forward-flow endpoints."""
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return cv2.remap(
        image,
        xx + flow[..., 0],
        yy + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )


def robust_motion_calibration(depth, motion, poses, intrinsics):
    """Infer Isaac MV offset/sign/scale from the dominant static scene."""
    n_frames = depth.shape[0]
    candidates = []
    step = max(1, (n_frames - 1) // 24)
    for offset in (0, 1):
        for sign in (-1.0, 1.0):
            observed_samples = []
            predicted_samples = []
            for index in range(0, n_frames - 1, step):
                predicted, _, valid = ego_geometry(
                    depth[index], poses[index], poses[index + 1], intrinsics[index]
                )
                observed = sign * np.asarray(
                    motion[min(index + offset, n_frames - 1)], dtype=np.float32
                )
                keep = (
                    valid
                    & np.isfinite(observed).all(axis=-1)
                    & ~depth_edge_mask(depth[index], radius=1)
                )
                keep[1::2, :] = False
                keep[:, 1::2] = False
                if np.count_nonzero(keep) < 100:
                    continue
                observed_samples.append(observed[keep])
                predicted_samples.append(predicted[keep])
            if not observed_samples:
                continue
            observed = np.concatenate(observed_samples)
            predicted = np.concatenate(predicted_samples)
            power = np.sum(observed * observed, axis=-1)
            usable = power > 1e-6
            scale = 1.0
            for _ in range(5):
                error = np.linalg.norm(scale * observed - predicted, axis=-1)
                cutoff = float(np.quantile(error[usable], 0.70))
                inliers = usable & (error <= cutoff)
                denominator = float(np.sum(power[inliers]))
                if denominator <= 1e-8:
                    break
                scale = float(
                    np.sum(observed[inliers] * predicted[inliers]) / denominator
                )
                scale = float(np.clip(scale, 0.25, 4.0))
            error = np.linalg.norm(scale * observed - predicted, axis=-1)
            candidates.append(
                {
                    "offset": offset,
                    "sign": sign,
                    "scale": scale,
                    "median_dominant_error_px": float(np.median(error)),
                    "p90_dominant_error_px": float(np.quantile(error, 0.90)),
                }
            )
    if not candidates:
        raise RuntimeError("could not calibrate motion-vector convention")
    best = dict(min(candidates, key=lambda item: item["median_dominant_error_px"]))
    best["candidates"] = [dict(item) for item in candidates]
    return best


def _gray_log(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.log1p(gray)


def _chromaticity(rgb):
    values = rgb.astype(np.float32) + 1.0
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-6)


def _dilate_float(values, radius):
    if radius <= 0:
        return values
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(values.astype(np.float32), kernel)


def _robust_threshold(values, valid, quantile, minimum):
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        return float(minimum)
    return max(float(minimum), float(np.quantile(selected, quantile)))


def fuse_motion_lighting(
    flow_confidence,
    depth_confidence,
    photo_confidence,
    chroma_confidence,
    persistent,
    valid,
):
    """Fuse motion and lighting cues without semantic or object-state input."""
    geometry_support = np.maximum.reduce(
        [depth_confidence, chroma_confidence, 0.25 * persistent]
    )
    illumination_confidence = (
        photo_confidence * (1.0 - geometry_support) * valid
    )
    dynamic_confidence = (
        flow_confidence * (1.0 - 0.85 * illumination_confidence)
    )
    static_confidence = (
        (1.0 - flow_confidence)
        * (1.0 - illumination_confidence)
        * valid
    )
    unknown_confidence = np.maximum(
        illumination_confidence,
        1.0 - np.maximum(dynamic_confidence, static_confidence),
    )
    return (
        static_confidence,
        dynamic_confidence,
        illumination_confidence,
        unknown_confidence,
    )


def classify_maps(
    rgb,
    depth,
    motion,
    poses,
    intrinsics,
    calibration,
    flow_quantile=0.95,
    depth_quantile=0.95,
    photo_quantile=0.95,
):
    """Build semantic-free static/dynamic/illumination confidence maps."""
    n_frames, height, width = depth.shape
    shape = (n_frames, height, width)
    ego_maps = np.zeros((*shape, 2), np.float16)
    flow_residual = np.zeros(shape, np.float16)
    depth_residual = np.zeros(shape, np.float16)
    photo_residual = np.zeros(shape, np.float16)
    chroma_residual = np.zeros(shape, np.float16)
    valid_maps = np.zeros(shape, bool)
    reliable_maps = np.zeros(shape, bool)
    offset = int(calibration["offset"])
    sign = float(calibration["sign"])
    scale = float(calibration["scale"])

    for index in range(n_frames - 1):
        ego, predicted_depth, valid = ego_geometry(
            depth[index], poses[index], poses[index + 1], intrinsics[index]
        )
        observed = scale * sign * np.asarray(
            motion[min(index + offset, n_frames - 1)], dtype=np.float32
        )
        flow_error = np.linalg.norm(observed - ego, axis=-1)
        sampled_depth = _sample_forward(depth[index + 1], ego)
        relative_depth_error = np.abs(sampled_depth - predicted_depth) / np.maximum(
            predicted_depth, 0.05
        )

        log0 = _gray_log(rgb[index])
        sampled_log1 = _sample_forward(_gray_log(rgb[index + 1]), ego)
        log_delta = sampled_log1 - log0
        base = (
            valid
            & np.isfinite(relative_depth_error)
            & (relative_depth_error < 0.03)
            & ~depth_edge_mask(depth[index], radius=1)
        )
        exposure_shift = float(np.median(log_delta[base])) if np.any(base) else 0.0
        light_error = np.abs(log_delta - exposure_shift)
        chroma0 = _chromaticity(rgb[index])
        sampled_chroma1 = _sample_forward(_chromaticity(rgb[index + 1]), ego)
        color_error = np.linalg.norm(sampled_chroma1 - chroma0, axis=-1)

        valid &= (
            np.isfinite(flow_error)
            & np.isfinite(relative_depth_error)
            & np.isfinite(light_error)
            & np.isfinite(color_error)
        )
        reliable = valid & ~depth_edge_mask(depth[index], radius=2)
        flow_error[~valid] = 0.0
        relative_depth_error[~valid] = 0.0
        light_error[~valid] = 0.0
        color_error[~valid] = 0.0
        ego_maps[index] = ego.astype(np.float16)
        flow_residual[index] = flow_error.astype(np.float16)
        depth_residual[index] = relative_depth_error.astype(np.float16)
        photo_residual[index] = light_error.astype(np.float16)
        chroma_residual[index] = color_error.astype(np.float16)
        valid_maps[index] = valid
        reliable_maps[index] = reliable

    for values in (
        ego_maps,
        flow_residual,
        depth_residual,
        photo_residual,
        chroma_residual,
        valid_maps,
        reliable_maps,
    ):
        values[-1] = values[-2]

    flow_threshold = _robust_threshold(
        flow_residual.astype(np.float32), reliable_maps, flow_quantile, 0.35
    )
    ego_magnitude = np.linalg.norm(ego_maps.astype(np.float32), axis=-1)
    relative_flow_residual = (
        flow_residual.astype(np.float32) / (1.0 + ego_magnitude)
    )
    relative_values = relative_flow_residual[
        reliable_maps & np.isfinite(relative_flow_residual)
    ]
    relative_flow_threshold = float(
        np.clip(2.5 * np.quantile(relative_values, 0.80), 0.18, 0.40)
    )
    presumed_static = (
        reliable_maps
        & (flow_residual.astype(np.float32) <= flow_threshold)
        & (relative_flow_residual <= relative_flow_threshold)
    )
    depth_threshold = _robust_threshold(
        depth_residual.astype(np.float32),
        presumed_static,
        depth_quantile,
        0.015,
    )
    photo_threshold = _robust_threshold(
        photo_residual.astype(np.float32),
        presumed_static
        & (depth_residual.astype(np.float32) <= depth_threshold),
        photo_quantile,
        0.04,
    )
    chroma_threshold = _robust_threshold(
        chroma_residual.astype(np.float32),
        presumed_static
        & (depth_residual.astype(np.float32) <= depth_threshold),
        photo_quantile,
        0.015,
    )

    absolute_flow_confidence = _sigmoid(
        (flow_residual.astype(np.float32) - flow_threshold)
        / max(0.12, 0.25 * flow_threshold)
    ) * reliable_maps
    relative_flow_confidence = _sigmoid(
        (relative_flow_residual - relative_flow_threshold)
        / max(0.04, 0.25 * relative_flow_threshold)
    ) * reliable_maps
    flow_confidence = np.maximum(
        absolute_flow_confidence, relative_flow_confidence
    )
    depth_confidence = _sigmoid(
        (depth_residual.astype(np.float32) - depth_threshold)
        / max(0.006, 0.35 * depth_threshold)
    ) * valid_maps
    photo_confidence = _sigmoid(
        (photo_residual.astype(np.float32) - photo_threshold)
        / max(0.02, 0.35 * photo_threshold)
    ) * valid_maps
    chroma_confidence = _sigmoid(
        (chroma_residual.astype(np.float32) - chroma_threshold)
        / max(0.008, 0.35 * chroma_threshold)
    ) * valid_maps

    for index in range(n_frames):
        spread = _dilate_float(flow_confidence[index], radius=2)
        on_edge = valid_maps[index] & ~reliable_maps[index]
        flow_confidence[index][on_edge] = spread[on_edge]
        depth_confidence[index] = (
            _dilate_float(depth_confidence[index], radius=2) * valid_maps[index]
        )
        chroma_confidence[index] = (
            _dilate_float(chroma_confidence[index], radius=1) * valid_maps[index]
        )

    persistent = np.zeros(shape, np.float32)
    for index in range(n_frames):
        neighbors = []
        if index > 0:
            neighbors.append(_dilate_float(flow_confidence[index - 1], radius=3))
        if index + 1 < n_frames:
            neighbors.append(_dilate_float(flow_confidence[index + 1], radius=3))
        neighbor = (
            np.maximum.reduce(neighbors)
            if neighbors
            else flow_confidence[index]
        )
        persistent[index] = np.sqrt(
            np.clip(flow_confidence[index] * neighbor, 0, 1)
        )

    (
        static_confidence,
        dynamic_confidence,
        illumination_confidence,
        unknown_confidence,
    ) = fuse_motion_lighting(
        flow_confidence,
        depth_confidence,
        photo_confidence,
        chroma_confidence,
        persistent,
        valid_maps,
    )
    maps = {
        "ego_flow_px": ego_maps,
        "flow_residual_px": flow_residual,
        "flow_residual_relative": relative_flow_residual.astype(np.float16),
        "depth_residual_relative": depth_residual,
        "photo_residual_log": photo_residual,
        "chroma_residual": chroma_residual,
        "valid": valid_maps.astype(np.uint8),
        "q_motion_u8": np.round(
            np.clip(flow_confidence, 0, 1) * 255
        ).astype(np.uint8),
        "q_static_u8": np.round(np.clip(static_confidence, 0, 1) * 255).astype(np.uint8),
        "q_dynamic_u8": np.round(np.clip(dynamic_confidence, 0, 1) * 255).astype(np.uint8),
        "q_illumination_u8": np.round(
            np.clip(illumination_confidence, 0, 1) * 255
        ).astype(np.uint8),
        "q_unknown_u8": np.round(
            np.clip(unknown_confidence, 0, 1) * 255
        ).astype(np.uint8),
    }
    thresholds = {
        "flow_threshold_px": float(flow_threshold),
        "relative_flow_threshold": relative_flow_threshold,
        "depth_threshold_relative": float(depth_threshold),
        "photo_threshold_log": float(photo_threshold),
        "chroma_threshold": float(chroma_threshold),
    }
    return maps, thresholds


def _sample_event_map(maps, frame_index, x, y):
    height, width = maps.shape[1:3]
    result = np.zeros(x.shape, np.float16)
    inside = (x < width) & (y < height)
    indices = np.flatnonzero(inside)
    result[indices] = (
        maps[frame_index[indices], y[indices], x[indices]].astype(np.float32)
        / 255.0
    ).astype(np.float16)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--camera", default="wrist_cam")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--event-window-ms", type=float, default=10.0)
    parser.add_argument("--flow-quantile", type=float, default=0.95)
    parser.add_argument("--depth-quantile", type=float, default=0.95)
    parser.add_argument("--photo-quantile", type=float, default=0.95)
    parser.add_argument("--event-confidence-threshold", type=float, default=0.50)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.camera

    with h5py.File(args.episode, "r") as episode:
        required = {
            f"{prefix}_rgb",
            f"{prefix}_depth_metric",
            f"{prefix}_motion_vectors",
            f"{prefix}_pose_w_ros",
            f"{prefix}_intrinsics",
        }
        missing = sorted(required - set(episode.keys()))
        if missing:
            raise SystemExit(
                "episode lacks motion-separation geometry; regenerate with "
                "--event_dynamic_gt: " + ", ".join(missing)
            )
        rgb = episode[f"{prefix}_rgb"][:]
        depth = episode[f"{prefix}_depth_metric"][:].astype(np.float32).squeeze(-1)
        motion = episode[f"{prefix}_motion_vectors"][:].astype(np.float32)
        poses = episode[f"{prefix}_pose_w_ros"][:].astype(np.float64)
        intrinsics = episode[f"{prefix}_intrinsics"][:].astype(np.float64)
        segmentation = (
            episode[f"{prefix}_seg"][:].squeeze(-1)
            if f"{prefix}_seg" in episode
            else None
        )
        object_speed = (
            np.linalg.norm(episode["object_vel"][:], axis=-1)
            if "object_vel" in episode
            else None
        )

    calibration = robust_motion_calibration(depth, motion, poses, intrinsics)
    maps, thresholds = classify_maps(
        rgb,
        depth,
        motion,
        poses,
        intrinsics,
        calibration,
        flow_quantile=args.flow_quantile,
        depth_quantile=args.depth_quantile,
        photo_quantile=args.photo_quantile,
    )
    n_frames, height, width = depth.shape
    with h5py.File(args.events, "r") as event_file:
        root = event_file[f"DVS/{args.camera}"]
        x = root["x"][:]
        y = root["y"][:]
        timestamps = root["t"][:]
        polarity = root["p"][:]
        quality = root["q"][:] if "q" in root else np.ones(x.shape, np.float16)
        time_origin = float(
            event_file.attrs.get("event_time_origin_s", timestamps.min())
        )

    frame_dt = 1.0 / float(args.fps)
    frame_index = np.floor((timestamps - time_origin) / frame_dt).astype(np.int64)
    frame_index = np.clip(frame_index, 0, n_frames - 1)
    q_static = _sample_event_map(maps["q_static_u8"], frame_index, x, y)
    q_motion = _sample_event_map(maps["q_motion_u8"], frame_index, x, y)
    q_dynamic = _sample_event_map(maps["q_dynamic_u8"], frame_index, x, y)
    q_illumination = _sample_event_map(
        maps["q_illumination_u8"], frame_index, x, y
    )
    q_unknown = _sample_event_map(maps["q_unknown_u8"], frame_index, x, y)

    derived_h5 = args.out_dir / f"{args.camera}_motion_separated_events.h5"
    with h5py.File(derived_h5, "w") as output:
        output.attrs["schema_version"] = 2
        output.attrs["algorithm"] = "lighting_suppressed_ego_motion_v2"
        output.attrs["source_events"] = str(args.events.resolve())
        output.attrs["source_episode"] = str(args.episode.resolve())
        output.attrs["event_time_origin_s"] = time_origin
        for key, value in thresholds.items():
            output.attrs[key] = value
        output.attrs["mv_offset"] = int(calibration["offset"])
        output.attrs["mv_sign"] = float(calibration["sign"])
        output.attrs["mv_scale"] = float(calibration["scale"])
        group = output.create_group(f"DVS/{args.camera}")
        for name, values in (
            ("x", x),
            ("y", y),
            ("t", timestamps),
            ("p", polarity),
            ("q", quality),
            ("q_motion", q_motion),
            ("q_static", q_static),
            ("q_dynamic", q_dynamic),
            ("q_illumination", q_illumination),
            ("q_unknown", q_unknown),
        ):
            _write_dataset(group, name, values)
        map_group = output.create_group("MotionSeparation")
        for name, values in maps.items():
            _write_dataset(map_group, name, values)

    event_threshold = float(args.event_confidence_threshold)
    inside = (x < width) & (y < height)
    indices = np.flatnonzero(inside)
    motion_candidates = q_motion[indices] >= event_threshold
    dynamic_retained = q_dynamic[indices] >= event_threshold
    metrics = {
        "schema_version": 2,
        "algorithm": "lighting_suppressed_ego_motion_v2",
        "algorithm_uses_semantics": False,
        "algorithm_uses_object_velocity": False,
        "camera": args.camera,
        "frames": int(n_frames),
        "events": int(x.size),
        "motion_vector_calibration": calibration,
        **thresholds,
        "event_confidence_threshold": event_threshold,
        "motion_candidate_event_fraction": float(np.mean(motion_candidates)),
        "static_event_fraction": float(np.mean(q_static[indices] >= event_threshold)),
        "dynamic_event_fraction": float(np.mean(dynamic_retained)),
        "illumination_rejected_motion_fraction": (
            float(np.mean(motion_candidates & ~dynamic_retained))
            / max(float(np.mean(motion_candidates)), 1e-12)
        ),
        "illumination_event_fraction": float(
            np.mean(q_illumination[indices] >= event_threshold)
        ),
        "unknown_event_fraction": float(np.mean(q_unknown[indices] >= event_threshold)),
        "derived_h5": str(derived_h5),
    }
    if segmentation is not None:
        event_labels = segmentation[
            frame_index[indices], y[indices], x[indices]
        ]
        retained = q_dynamic[indices].astype(np.float32) >= event_threshold
        for name, label_mask in (
            ("background", np.isin(event_labels, (0, 3))),
            ("robot", event_labels == 1),
            ("target", event_labels == 2),
        ):
            metrics[f"eval_{name}_dynamic_retention"] = (
                float(np.mean(retained[label_mask]))
                if np.any(label_mask)
                else None
            )
        metrics["evaluation_uses_semantics"] = True
        if object_speed is not None:
            moving_target = (
                (segmentation[:-1] == 2)
                & (object_speed[:-1, None, None] > 0.02)
                & maps["valid"][:-1].astype(bool)
            )
            dynamic_pixels = (
                maps["q_dynamic_u8"][:-1].astype(np.float32) / 255.0
                >= event_threshold
            )
            metrics["eval_moving_target_pixel_recall"] = (
                float(np.mean(dynamic_pixels[moving_target]))
                if np.any(moving_target)
                else None
            )
            metrics["evaluation_uses_object_velocity"] = True

    metrics_path = args.out_dir / f"{args.camera}_motion_separation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    comparison_path = args.out_dir / f"{args.camera}_motion_separation.mp4"
    diagnostic_path = args.out_dir / f"{args.camera}_motion_diagnostic.mp4"
    static_path = args.out_dir / f"{args.camera}_static_events.mp4"
    dynamic_path = args.out_dir / f"{args.camera}_dynamic_events.mp4"
    illumination_path = args.out_dir / f"{args.camera}_illumination_events.mp4"
    comparison_writer = H264Writer(str(comparison_path), width * 4, height, args.fps)
    diagnostic_writer = H264Writer(
        str(diagnostic_path), width * 3, height * 2, args.fps
    )
    static_writer = H264Writer(str(static_path), width, height, args.fps)
    dynamic_writer = H264Writer(str(dynamic_path), width, height, args.fps)
    illumination_writer = H264Writer(
        str(illumination_path), width, height, args.fps
    )
    event_window = float(args.event_window_ms) * 1e-3
    ego_magnitude = np.linalg.norm(maps["ego_flow_px"].astype(np.float32), axis=-1)
    ego_max = max(1.0, float(np.quantile(ego_magnitude, 0.99)))
    for index in range(n_frames):
        t0 = time_origin + index * frame_dt
        lo = int(np.searchsorted(timestamps, t0, side="left"))
        hi = int(np.searchsorted(timestamps, t0 + event_window, side="left"))
        xi, yi, pi = x[lo:hi], y[lo:hi], polarity[lo:hi]
        raw_image = _event_image(xi, yi, pi, height, width)
        static_image = _event_image(
            xi, yi, pi, height, width, q_static[lo:hi], minimum=event_threshold
        )
        dynamic_image = _event_image(
            xi, yi, pi, height, width, q_dynamic[lo:hi], minimum=event_threshold
        )
        illumination_image = _event_image(
            xi, yi, pi, height, width, q_illumination[lo:hi], minimum=event_threshold
        )
        comparison_writer.write(
            np.concatenate(
                [
                    _label(raw_image, "Raw Event"),
                    _label(static_image, "Static / Ego-consistent"),
                    _label(dynamic_image, "Dynamic / Geometry-supported"),
                    _label(illumination_image, "Illumination / Unknown"),
                ],
                axis=1,
            )
        )
        static_writer.write(static_image)
        dynamic_writer.write(dynamic_image)
        illumination_writer.write(illumination_image)
        valid = maps["valid"][index].astype(bool)
        diagnostic_writer.write(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            _label(cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR), "RGB"),
                            _label(
                                _flow_image(
                                    maps["ego_flow_px"][index].astype(np.float32),
                                    valid,
                                    ego_max,
                                ),
                                "Predicted Ego Flow",
                            ),
                            _label(raw_image, "Raw Event"),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            _label(
                                _heatmap(
                                    maps["flow_residual_px"][index].astype(np.float32),
                                    3.0 * thresholds["flow_threshold_px"],
                                    valid,
                                ),
                                "Flow residual",
                            ),
                            _label(dynamic_image, "Dynamic"),
                            _label(illumination_image, "Illumination / Unknown"),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
        )

    for writer in (
        comparison_writer,
        diagnostic_writer,
        static_writer,
        dynamic_writer,
        illumination_writer,
    ):
        writer.release()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[MotionSeparation] comparison -> {comparison_path}")
    print(f"[MotionSeparation] diagnostic -> {diagnostic_path}")


if __name__ == "__main__":
    main()
