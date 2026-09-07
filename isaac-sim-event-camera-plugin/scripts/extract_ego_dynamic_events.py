#!/usr/bin/env python3
"""Extract ego-compensated dynamic wrist-camera events from a DOM episode.

The script compares Isaac's observed screen-space motion vectors with the
static-scene flow predicted from metric depth and consecutive camera poses.  It
produces soft per-pixel/per-event dynamic confidence, a semantic target-dynamic
confidence, derived HDF5 data, quantitative diagnostics, and H.264 videos.

Dense geometry is expected in the episode HDF5 under these keys (recorded by
DynamicVLA's ``--event_dynamic_gt`` flag):

``wrist_cam_depth_metric``, ``wrist_cam_motion_vectors``,
``wrist_cam_pose_w_ros``, and ``wrist_cam_intrinsics``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from dvs_gen.io import H264Writer


def _rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def ego_flow(
    depth: np.ndarray,
    pose0: np.ndarray,
    pose1: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return static-world forward flow and its geometric validity mask."""
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    z = depth
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    points_c0 = np.stack([x, y, z], axis=-1)

    r0 = _rotation_wxyz(pose0[3:])
    r1 = _rotation_wxyz(pose1[3:])
    points_w = points_c0 @ r0.T + pose0[:3]
    points_c1 = (points_w - pose1[:3]) @ r1
    z1 = points_c1[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u1 = fx * points_c1[..., 0] / z1 + cx
        v1 = fy * points_c1[..., 1] / z1 + cy
    flow = np.stack([u1 - xx, v1 - yy], axis=-1).astype(np.float32)
    valid = (
        np.isfinite(depth)
        & np.isfinite(flow).all(axis=-1)
        & (depth > 0.02)
        & (depth < 100.0)
        & (z1 > 0.02)
        # Numerical round-off can move an identity-projected border pixel a
        # few ulps outside zero. Keep a small projection tolerance.
        & (u1 >= -1e-4)
        & (u1 <= width - 1 + 1e-4)
        & (v1 >= -1e-4)
        & (v1 <= height - 1 + 1e-4)
    )
    flow[~valid] = 0.0
    return flow, valid


def depth_edge_mask(depth: np.ndarray, radius: int = 2) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    dx = np.zeros_like(depth)
    dy = np.zeros_like(depth)
    dx[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
    dy[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
    tolerance = 0.03 + 0.04 * np.clip(depth, 0.0, 10.0)
    edge = (np.maximum(dx, dy) > tolerance).astype(np.uint8)
    if radius > 0:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        edge = cv2.dilate(edge, kernel)
    return edge.astype(bool)


def _robust_motion_calibration(
    depth: np.ndarray,
    motion: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    segmentation: np.ndarray,
) -> dict:
    """Choose MV temporal anchor/sign/scale on known-static simulation pixels."""
    n_frames = depth.shape[0]
    candidates = []
    for offset in (0, 1):
        for sign in (-1.0, 1.0):
            mv_samples = []
            ego_samples = []
            for index in range(0, n_frames - 1, max(1, (n_frames - 1) // 24)):
                mv_index = min(index + offset, n_frames - 1)
                predicted, valid = ego_flow(
                    depth[index], poses[index], poses[index + 1], intrinsics[index]
                )
                static = np.isin(segmentation[index].squeeze(), (0, 3))
                keep = valid & static & ~depth_edge_mask(depth[index], radius=1)
                keep[1::2, :] = False
                keep[:, 1::2] = False
                observed = sign * np.asarray(motion[mv_index], dtype=np.float32)
                finite = keep & np.isfinite(observed).all(axis=-1)
                if np.count_nonzero(finite) < 100:
                    continue
                mv_samples.append(observed[finite])
                ego_samples.append(predicted[finite])
            if not mv_samples:
                continue
            mv_flat = np.concatenate(mv_samples, axis=0)
            ego_flat = np.concatenate(ego_samples, axis=0)
            denominator = float(np.sum(mv_flat * mv_flat))
            scale = float(np.sum(mv_flat * ego_flat) / max(denominator, 1e-8))
            scale = float(np.clip(scale, 0.25, 4.0))
            error = np.linalg.norm(scale * mv_flat - ego_flat, axis=-1)
            candidates.append(
                {
                    "offset": offset,
                    "sign": sign,
                    "scale": scale,
                    "median_static_error_px": float(np.median(error)),
                    "p90_static_error_px": float(np.quantile(error, 0.90)),
                }
            )
    if not candidates:
        raise RuntimeError("could not calibrate motion-vector convention")
    best = dict(min(candidates, key=lambda item: item["median_static_error_px"]))
    best["candidates"] = [dict(item) for item in candidates]
    return best


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-values))


def _flow_image(flow: np.ndarray, valid: np.ndarray, magnitude_max: float) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*magnitude.shape, 3), np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / max(magnitude_max, 1e-6) * 255, 0, 255).astype(np.uint8)
    hsv[~valid] = 0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _heatmap(values: np.ndarray, maximum: float, valid: np.ndarray) -> np.ndarray:
    scaled = np.clip(values / max(maximum, 1e-6) * 255, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def _event_image(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    height: int,
    width: int,
    weights: np.ndarray | None = None,
    minimum: float = 0.0,
) -> np.ndarray:
    image = np.full((height, width, 3), 255, np.uint8)
    keep = (x < width) & (y < height)
    if weights is not None:
        keep &= weights >= minimum
    x = x[keep].astype(np.intp)
    y = y[keep].astype(np.intp)
    polarity = polarity[keep]
    if weights is None:
        strength = np.ones(x.shape[0], np.float32)
    else:
        strength = np.clip(weights[keep].astype(np.float32), 0.0, 1.0)
    faded = (255.0 * (1.0 - strength)).astype(np.uint8)
    on = polarity > 0
    image[y[on], x[on], 0] = faded[on]
    image[y[on], x[on], 1] = faded[on]
    image[y[on], x[on], 2] = 255
    image[y[~on], x[~on], 0] = 255
    image[y[~on], x[~on], 1] = faded[~on]
    image[y[~on], x[~on], 2] = faded[~on]
    return image


def _label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 330), 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _write_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    group.create_dataset(name, data=data, compression="gzip", compression_opts=4, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--camera", default="wrist_cam")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--event-window-ms", type=float, default=10.0)
    parser.add_argument("--minimum-threshold-px", type=float, default=0.35)
    parser.add_argument("--static-quantile", type=float, default=0.99)
    parser.add_argument("--memory-decay", type=float, default=0.85)
    parser.add_argument("--event-confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.camera
    with h5py.File(args.episode, "r") as episode:
        required = {
            f"{prefix}_rgb",
            f"{prefix}_seg",
            f"{prefix}_depth_metric",
            f"{prefix}_motion_vectors",
            f"{prefix}_pose_w_ros",
            f"{prefix}_intrinsics",
        }
        missing = sorted(required - set(episode.keys()))
        if missing:
            raise SystemExit(
                "episode lacks ECDM geometry; regenerate with --event_dynamic_gt: "
                + ", ".join(missing)
            )
        rgb = episode[f"{prefix}_rgb"][:]
        segmentation = episode[f"{prefix}_seg"][:]
        depth = episode[f"{prefix}_depth_metric"][:].astype(np.float32)
        motion = episode[f"{prefix}_motion_vectors"][:].astype(np.float32)
        poses = episode[f"{prefix}_pose_w_ros"][:].astype(np.float64)
        intrinsics = episode[f"{prefix}_intrinsics"][:].astype(np.float64)
        object_speed = np.linalg.norm(episode["object_vel"][:], axis=-1)

    n_frames, height, width = rgb.shape[:3]
    calibration = _robust_motion_calibration(
        depth, motion, poses, intrinsics, segmentation
    )
    offset = int(calibration["offset"])
    sign = float(calibration["sign"])
    scale = float(calibration["scale"])

    ego_maps = np.zeros((n_frames, height, width, 2), np.float16)
    residual_maps = np.zeros((n_frames, height, width), np.float16)
    valid_maps = np.zeros((n_frames, height, width), bool)
    residual_static_samples = []
    for index in range(n_frames - 1):
        ego, valid = ego_flow(depth[index], poses[index], poses[index + 1], intrinsics[index])
        mv_index = min(index + offset, n_frames - 1)
        observed = scale * sign * motion[mv_index]
        residual = np.linalg.norm(observed - ego, axis=-1)
        valid &= np.isfinite(residual) & ~depth_edge_mask(depth[index], radius=2)
        residual[~valid] = 0.0
        ego_maps[index] = ego.astype(np.float16)
        residual_maps[index] = residual.astype(np.float16)
        valid_maps[index] = valid
        static = np.isin(segmentation[index].squeeze(), (0, 3)) & valid
        if np.any(static):
            sample = residual[static]
            stride = max(1, sample.size // 100_000)
            residual_static_samples.append(sample[::stride])
    ego_maps[-1] = ego_maps[-2]
    residual_maps[-1] = residual_maps[-2]
    valid_maps[-1] = valid_maps[-2]

    static_values = np.concatenate(residual_static_samples)
    threshold = max(
        float(args.minimum_threshold_px),
        float(np.quantile(static_values, args.static_quantile)),
    )
    softness = max(0.08, 0.25 * threshold)
    dynamic_maps = np.zeros((n_frames, height, width), np.uint8)
    target_maps = np.zeros_like(dynamic_maps)
    target_memory = np.zeros((height, width), np.float32)
    for index in range(n_frames):
        residual = residual_maps[index].astype(np.float32)
        confidence = _sigmoid((residual - threshold) / softness)
        confidence *= valid_maps[index]
        dynamic_maps[index] = np.round(confidence * 255).astype(np.uint8)

        target = segmentation[index].squeeze() == 2
        current_target = confidence * target
        target_memory = np.maximum(current_target, args.memory_decay * target_memory)
        # Semantic support relocates memory with the object instead of leaving
        # an image-space trail as the wrist camera moves.
        target_memory *= target
        target_maps[index] = np.round(target_memory * 255).astype(np.uint8)

    with h5py.File(args.events, "r") as event_file:
        root = event_file[f"DVS/{args.camera}"]
        x = root["x"][:]
        y = root["y"][:]
        timestamps = root["t"][:]
        polarity = root["p"][:]
        quality = root["q"][:] if "q" in root else np.ones(x.shape, np.float16)
        time_origin = float(event_file.attrs.get("event_time_origin_s", timestamps.min()))

    frame_dt = 1.0 / float(args.fps)
    frame_index = np.floor((timestamps - time_origin) / frame_dt).astype(np.int64)
    frame_index = np.clip(frame_index, 0, n_frames - 1)
    inside = (x < width) & (y < height)
    q_dynamic = np.zeros(x.shape, np.float16)
    q_target = np.zeros(x.shape, np.float16)
    valid_event_indices = np.flatnonzero(inside)
    q_dynamic[valid_event_indices] = (
        dynamic_maps[
            frame_index[valid_event_indices],
            y[valid_event_indices],
            x[valid_event_indices],
        ].astype(np.float32)
        / 255.0
    ).astype(np.float16)
    q_target[valid_event_indices] = (
        target_maps[
            frame_index[valid_event_indices],
            y[valid_event_indices],
            x[valid_event_indices],
        ].astype(np.float32)
        / 255.0
    ).astype(np.float16)

    derived_h5 = args.out_dir / f"{args.camera}_ecdm_events.h5"
    with h5py.File(derived_h5, "w") as output:
        output.attrs["source_events"] = str(args.events.resolve())
        output.attrs["source_episode"] = str(args.episode.resolve())
        output.attrs["event_time_origin_s"] = time_origin
        output.attrs["ecdm_mv_offset"] = offset
        output.attrs["ecdm_mv_sign"] = sign
        output.attrs["ecdm_mv_scale"] = scale
        output.attrs["ecdm_residual_threshold_px"] = threshold
        output.attrs["ecdm_memory_decay"] = float(args.memory_decay)
        group = output.create_group(f"DVS/{args.camera}")
        for name, values in (
            ("x", x),
            ("y", y),
            ("t", timestamps),
            ("p", polarity),
            ("q", quality),
            ("q_dyn", q_dynamic),
            ("q_target_dyn", q_target),
        ):
            _write_dataset(group, name, values)
        maps = output.create_group("ECDM")
        _write_dataset(maps, "dynamic_confidence_u8", dynamic_maps)
        _write_dataset(maps, "target_dynamic_confidence_u8", target_maps)
        _write_dataset(maps, "residual_flow_px", residual_maps)
        _write_dataset(maps, "ego_flow_px", ego_maps)
        _write_dataset(maps, "valid", valid_maps.astype(np.uint8))

    threshold_event = float(args.event_confidence_threshold)
    event_seg = segmentation[
        frame_index[valid_event_indices],
        y[valid_event_indices],
        x[valid_event_indices],
        0,
    ]
    qd_valid = q_dynamic[valid_event_indices].astype(np.float32)
    retained = qd_valid >= threshold_event
    background = np.isin(event_seg, (0, 3))
    target = event_seg == 2
    robot = event_seg == 1
    dynamic_target_pixels = (
        (segmentation[:-1, ..., 0] == 2)
        & (object_speed[:-1, None, None] > 0.02)
        & valid_maps[:-1]
    )
    target_detected_pixels = dynamic_maps[:-1] >= round(255 * threshold_event)
    target_pixel_recall = float(
        np.count_nonzero(target_detected_pixels & dynamic_target_pixels)
        / max(np.count_nonzero(dynamic_target_pixels), 1)
    )

    metrics = {
        "schema_version": 1,
        "camera": args.camera,
        "frames": n_frames,
        "events": int(x.size),
        "motion_vector_calibration": calibration,
        "residual_threshold_px": threshold,
        "residual_softness_px": softness,
        "static_residual_median_px": float(np.median(static_values)),
        "static_residual_p95_px": float(np.quantile(static_values, 0.95)),
        "static_residual_p99_px": float(np.quantile(static_values, 0.99)),
        "dynamic_event_fraction": float(np.mean(q_dynamic >= threshold_event)),
        "target_dynamic_event_fraction": float(np.mean(q_target >= threshold_event)),
        "background_event_suppression": float(
            1.0 - np.count_nonzero(retained & background) / max(np.count_nonzero(background), 1)
        ),
        "target_event_retention": float(
            np.count_nonzero(retained & target) / max(np.count_nonzero(target), 1)
        ),
        "robot_event_retention": float(
            np.count_nonzero(retained & robot) / max(np.count_nonzero(robot), 1)
        ),
        "moving_target_pixel_recall": target_pixel_recall,
        "derived_h5": str(derived_h5.resolve()),
    }
    metrics_path = args.out_dir / f"{args.camera}_ecdm_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    six_panel_path = args.out_dir / f"{args.camera}_ecdm_six_panel.mp4"
    event_comparison_path = args.out_dir / f"{args.camera}_event_comparison.mp4"
    dynamic_path = args.out_dir / f"{args.camera}_dynamic_event.mp4"
    target_path = args.out_dir / f"{args.camera}_target_dynamic_event.mp4"
    six_writer = H264Writer(str(six_panel_path), width * 3, height * 2, args.fps)
    comparison_writer = H264Writer(
        str(event_comparison_path), width * 3, height, args.fps
    )
    dynamic_writer = H264Writer(str(dynamic_path), width, height, args.fps)
    target_writer = H264Writer(str(target_path), width, height, args.fps)
    event_window = float(args.event_window_ms) * 1e-3
    flow_max = max(1.0, float(np.quantile(np.linalg.norm(ego_maps.astype(np.float32), axis=-1), 0.99)))
    for index in range(n_frames):
        t0 = time_origin + index * frame_dt
        lo = int(np.searchsorted(timestamps, t0, side="left"))
        hi = int(np.searchsorted(timestamps, t0 + event_window, side="left"))
        xi, yi, pi = x[lo:hi], y[lo:hi], polarity[lo:hi]
        raw_image = _event_image(xi, yi, pi, height, width)
        dyn_image = _event_image(
            xi,
            yi,
            pi,
            height,
            width,
            q_dynamic[lo:hi],
            minimum=threshold_event,
        )
        target_image = _event_image(
            xi,
            yi,
            pi,
            height,
            width,
            q_target[lo:hi],
            minimum=threshold_event,
        )
        ego_image = _flow_image(
            ego_maps[index].astype(np.float32), valid_maps[index], flow_max
        )
        residual_image = _heatmap(
            residual_maps[index].astype(np.float32), 3.0 * threshold, valid_maps[index]
        )
        rgb_bgr = cv2.cvtColor(rgb[index, ..., :3], cv2.COLOR_RGB2BGR)
        top = np.concatenate(
            [
                _label(rgb_bgr, "RGB"),
                _label(raw_image, "Raw Event"),
                _label(ego_image, "Predicted Ego Flow"),
            ],
            axis=1,
        )
        bottom = np.concatenate(
            [
                _label(residual_image, f"Residual (tau={threshold:.2f}px)"),
                _label(dyn_image, "Dynamic-only Event"),
                _label(target_image, "Target-dynamic Event"),
            ],
            axis=1,
        )
        six_writer.write(np.concatenate([top, bottom], axis=0))
        comparison_writer.write(
            np.concatenate(
                [
                    _label(raw_image, "Raw Event"),
                    _label(dyn_image, "Dynamic-only Event"),
                    _label(target_image, "Target-dynamic Event"),
                ],
                axis=1,
            )
        )
        dynamic_writer.write(dyn_image)
        target_writer.write(target_image)
    six_writer.release()
    comparison_writer.release()
    dynamic_writer.release()
    target_writer.release()

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[ECDM] six-panel video -> {six_panel_path}", flush=True)
    print(f"[ECDM] event comparison video -> {event_comparison_path}", flush=True)
    print(f"[ECDM] dynamic event video -> {dynamic_path}", flush=True)
    print(f"[ECDM] target event video -> {target_path}", flush=True)


if __name__ == "__main__":
    main()
