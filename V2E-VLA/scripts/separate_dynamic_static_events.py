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


def voxelize_separated_events(
    x,
    y,
    timestamps,
    polarity,
    confidence_maps,
    *,
    start_time,
    num_bins,
    bin_seconds,
    output_size=(96, 128),
    clip_count=8.0,
):
    """Fuse confidence sampling and static/dynamic event voxelization.

    The generic training adapter uses ``np.add.at`` once for every output
    stream.  Online, static and dynamic events share the same coordinates,
    time bins, polarity, and illumination gate.  Building their flattened
    indices once and accumulating with ``np.bincount`` avoids repeated indexed
    writes, which are especially costly in high-event-rate windows.
    """
    out_h, out_w = output_size
    voxel_shape = (int(num_bins), 2, int(out_h), int(out_w))
    output_size_flat = int(np.prod(voxel_shape))
    if len(timestamps) == 0:
        empty = np.zeros(voxel_shape, np.float32)
        return empty, empty.copy()

    x = np.asarray(x)
    y = np.asarray(y)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    polarity = np.asarray(polarity)
    map_h, map_w = confidence_maps["q_static"].shape
    time_bin = np.floor((timestamps - start_time) / bin_seconds).astype(np.int64)
    output_x = (x.astype(np.int64) * out_w) // map_w
    output_y = (y.astype(np.int64) * out_h) // map_h
    valid = (
        (time_bin >= 0)
        & (time_bin < num_bins)
        & (x >= 0)
        & (x < map_w)
        & (y >= 0)
        & (y < map_h)
        & (output_x >= 0)
        & (output_x < out_w)
        & (output_y >= 0)
        & (output_y < out_h)
    )
    if not np.any(valid):
        empty = np.zeros(voxel_shape, np.float32)
        return empty, empty.copy()

    xv = x[valid].astype(np.intp, copy=False)
    yv = y[valid].astype(np.intp, copy=False)
    flat_index = (
        ((time_bin[valid] * 2 + (polarity[valid] > 0)) * out_h + output_y[valid])
        * out_w
        + output_x[valid]
    )
    illumination_keep = 1.0 - confidence_maps["q_illumination"][yv, xv]
    static_weight = confidence_maps["q_static"][yv, xv] * illumination_keep
    dynamic_weight = confidence_maps["q_dynamic"][yv, xv] * illumination_keep
    static = np.bincount(
        flat_index, weights=static_weight, minlength=output_size_flat
    ).astype(np.float32, copy=False).reshape(voxel_shape)
    dynamic = np.bincount(
        flat_index, weights=dynamic_weight, minlength=output_size_flat
    ).astype(np.float32, copy=False).reshape(voxel_shape)
    if clip_count > 0:
        normalizer = np.float32(np.log1p(clip_count))
        np.minimum(static, clip_count, out=static)
        np.minimum(dynamic, clip_count, out=dynamic)
        np.log1p(static, out=static)
        np.log1p(dynamic, out=dynamic)
        static /= normalizer
        dynamic /= normalizer
    return static, dynamic


def voxelize_separated_events_torch(
    x,
    y,
    timestamps,
    polarity,
    confidence_maps,
    *,
    start_time,
    num_bins,
    bin_seconds,
    output_size=(96, 128),
    clip_count=8.0,
    device="cuda",
):
    """Torch/CUDA equivalent of :func:`voxelize_separated_events`.

    Importing torch is delayed so the NumPy/OpenCV streaming path keeps a
    lightweight startup.  Passing an unavailable CUDA device raises a clear
    error instead of silently moving a real-time pipeline back to CPU.
    """
    import torch

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA voxelization requested but CUDA is unavailable")

    def tensor(value, dtype=None):
        return torch.as_tensor(value, dtype=dtype, device=target)

    x = tensor(x, torch.long)
    y = tensor(y, torch.long)
    timestamps = tensor(timestamps, torch.float64)
    polarity = tensor(polarity)
    q_static = tensor(confidence_maps["q_static"], torch.float32)
    q_dynamic = tensor(confidence_maps["q_dynamic"], torch.float32)
    q_illumination = tensor(confidence_maps["q_illumination"], torch.float32)
    map_h, map_w = q_static.shape
    out_h, out_w = output_size
    voxel_shape = (int(num_bins), 2, int(out_h), int(out_w))
    output_size_flat = int(np.prod(voxel_shape))
    if timestamps.numel() == 0:
        empty = torch.zeros(voxel_shape, dtype=torch.float32, device=target)
        return empty, empty.clone()

    time_bin = torch.floor((timestamps - start_time) / bin_seconds).long()
    output_x = torch.div(x * out_w, map_w, rounding_mode="floor")
    output_y = torch.div(y * out_h, map_h, rounding_mode="floor")
    valid = (
        (time_bin >= 0)
        & (time_bin < num_bins)
        & (x >= 0)
        & (x < map_w)
        & (y >= 0)
        & (y < map_h)
        & (output_x >= 0)
        & (output_x < out_w)
        & (output_y >= 0)
        & (output_y < out_h)
    )
    x = x[valid]
    y = y[valid]
    flat_index = (
        ((time_bin[valid] * 2 + (polarity[valid] > 0)) * out_h + output_y[valid])
        * out_w
        + output_x[valid]
    )
    illumination_keep = 1.0 - q_illumination[y, x]
    static = torch.bincount(
        flat_index,
        weights=q_static[y, x] * illumination_keep,
        minlength=output_size_flat,
    ).reshape(voxel_shape)
    dynamic = torch.bincount(
        flat_index,
        weights=q_dynamic[y, x] * illumination_keep,
        minlength=output_size_flat,
    ).reshape(voxel_shape)
    if clip_count > 0:
        static.clamp_max_(clip_count).log1p_().div_(np.log1p(clip_count))
        dynamic.clamp_max_(clip_count).log1p_().div_(np.log1p(clip_count))
    return static, dynamic


class ReusableSeparatedEventVoxelizer:
    """Exact NumPy voxelizer with reusable event-sized work buffers.

    Raw camera streams have fixed image bounds and the timestamp ring already
    selects one exact interval.  ``assume_valid=True`` uses that contract to
    skip seven full event-array validation/filter passes.  Set it to false for
    imported or otherwise untrusted event files.
    """

    def __init__(
        self,
        source_size=(360, 480),
        output_size=(96, 128),
        initial_capacity=0,
    ):
        self.source_size = tuple(source_size)
        self.output_size = tuple(output_size)
        src_h, src_w = self.source_size
        out_h, out_w = self.output_size
        source_y, source_x = np.indices(self.source_size, dtype=np.int64)
        output_x = (source_x * out_w) // src_w
        output_y = (source_y * out_h) // src_h
        self._pixel_to_output = (output_y * out_w + output_x).ravel()
        self._capacity = 0
        self._source_index = None
        self._flat_index = None
        self._relative_time = None
        self._positive = None
        self._static_weight = None
        self._dynamic_weight = None
        self._illumination_keep = None
        self._ensure_capacity(int(initial_capacity))

    @property
    def capacity(self):
        return self._capacity

    def _ensure_capacity(self, required):
        if required <= self._capacity:
            return
        capacity = max(1024, self._capacity)
        while capacity < required:
            capacity *= 2
        self._capacity = capacity
        self._source_index = np.empty(capacity, np.int64)
        self._flat_index = np.empty(capacity, np.int64)
        self._relative_time = np.empty(capacity, np.float64)
        self._positive = np.empty(capacity, bool)
        self._static_weight = np.empty(capacity, np.float32)
        self._dynamic_weight = np.empty(capacity, np.float32)
        self._illumination_keep = np.empty(capacity, np.float32)

    def voxelize(
        self,
        x,
        y,
        timestamps,
        polarity,
        confidence_maps,
        *,
        start_time,
        num_bins,
        bin_seconds,
        clip_count=8.0,
        assume_valid=True,
    ):
        n_events = len(timestamps)
        voxel_shape = (int(num_bins), 2, *self.output_size)
        output_size_flat = int(np.prod(voxel_shape))
        if n_events == 0:
            empty = np.zeros(voxel_shape, np.float32)
            return empty, empty.copy()
        if not (len(x) == len(y) == len(polarity) == n_events):
            raise ValueError("x/y/timestamps/polarity must have equal lengths")
        self._ensure_capacity(n_events)
        source_index = self._source_index[:n_events]
        flat_index = self._flat_index[:n_events]
        relative_time = self._relative_time[:n_events]
        positive = self._positive[:n_events]
        static_weight = self._static_weight[:n_events]
        dynamic_weight = self._dynamic_weight[:n_events]
        illumination_keep = self._illumination_keep[:n_events]
        src_h, src_w = self.source_size
        out_h, out_w = self.output_size

        if not assume_valid:
            x_values = np.asarray(x)
            y_values = np.asarray(y)
            timestamp_values = np.asarray(timestamps)
            valid = (
                (x_values >= 0)
                & (x_values < src_w)
                & (y_values >= 0)
                & (y_values < src_h)
                & (timestamp_values >= start_time)
                & (timestamp_values < start_time + num_bins * bin_seconds)
            )
            if not np.all(valid):
                return voxelize_separated_events(
                    x_values,
                    y_values,
                    timestamp_values,
                    polarity,
                    confidence_maps,
                    start_time=start_time,
                    num_bins=num_bins,
                    bin_seconds=bin_seconds,
                    output_size=self.output_size,
                    clip_count=clip_count,
                )

        # Gather the three confidence maps while source_index still contains
        # source pixels; it is then safely reused as the temporal-bin buffer.
        # Force int64 arithmetic before writing the result.  The camera stores
        # coordinates as uint16, where ``y * width`` would otherwise overflow.
        np.multiply(
            y, src_w, out=source_index, dtype=np.int64, casting="unsafe"
        )
        np.add(
            source_index,
            x,
            out=source_index,
            dtype=np.int64,
            casting="unsafe",
        )
        np.take(
            confidence_maps["q_static"].reshape(-1),
            source_index,
            out=static_weight,
        )
        np.take(
            confidence_maps["q_dynamic"].reshape(-1),
            source_index,
            out=dynamic_weight,
        )
        np.take(
            confidence_maps["q_illumination"].reshape(-1),
            source_index,
            out=illumination_keep,
        )
        np.subtract(1.0, illumination_keep, out=illumination_keep)
        np.multiply(static_weight, illumination_keep, out=static_weight)
        np.multiply(dynamic_weight, illumination_keep, out=dynamic_weight)
        np.take(self._pixel_to_output, source_index, out=flat_index)

        np.subtract(timestamps, start_time, out=relative_time)
        np.divide(relative_time, bin_seconds, out=relative_time)
        np.floor(relative_time, out=relative_time)
        np.copyto(source_index, relative_time, casting="unsafe")
        np.greater(polarity, 0, out=positive)
        np.multiply(source_index, 2, out=source_index)
        np.add(source_index, positive, out=source_index, casting="unsafe")
        np.multiply(source_index, out_h * out_w, out=source_index)
        np.add(flat_index, source_index, out=flat_index)

        static = np.bincount(
            flat_index, weights=static_weight, minlength=output_size_flat
        ).astype(np.float32, copy=False).reshape(voxel_shape)
        dynamic = np.bincount(
            flat_index, weights=dynamic_weight, minlength=output_size_flat
        ).astype(np.float32, copy=False).reshape(voxel_shape)
        if clip_count > 0:
            normalizer = np.float32(np.log1p(clip_count))
            np.minimum(static, clip_count, out=static)
            np.minimum(dynamic, clip_count, out=dynamic)
            np.log1p(static, out=static)
            np.log1p(dynamic, out=dynamic)
            static /= normalizer
            dynamic /= normalizer
        return static, dynamic


class SeparatedEventVoxelRing:
    """Allocation-free chronological history for online event voxel bins.

    Every bin is mirrored into a second half of the buffer.  Consequently the
    latest history is always one contiguous slice, even after wrap-around; no
    ``np.roll`` or ``concatenate`` is needed before creating a torch tensor.
    Returned views remain valid until the next :meth:`append` call.
    """

    def __init__(self, history_bins=8, output_size=(96, 128), dtype=np.float32):
        self.history_bins = int(history_bins)
        self.output_size = tuple(output_size)
        if self.history_bins < 1:
            raise ValueError("history_bins must be positive")
        shape = (2 * self.history_bins, 2, *self.output_size)
        self.static = np.zeros(shape, dtype=dtype)
        self.dynamic = np.zeros(shape, dtype=dtype)
        self.cursor = 0
        self.filled = 0

    def reset(self):
        self.static.fill(0)
        self.dynamic.fill(0)
        self.cursor = 0
        self.filled = 0

    def append(self, static_bins, dynamic_bins):
        static_bins = np.asarray(static_bins, dtype=self.static.dtype)
        dynamic_bins = np.asarray(dynamic_bins, dtype=self.dynamic.dtype)
        expected_tail = (2, *self.output_size)
        if (
            static_bins.ndim != 4
            or dynamic_bins.shape != static_bins.shape
            or static_bins.shape[1:] != expected_tail
        ):
            raise ValueError(
                "static/dynamic bins must share shape [T,2,H,W] matching the ring"
            )
        # More than one full history only needs its newest bins.
        static_bins = static_bins[-self.history_bins :]
        dynamic_bins = dynamic_bins[-self.history_bins :]
        for static_bin, dynamic_bin in zip(static_bins, dynamic_bins):
            np.copyto(self.static[self.cursor], static_bin)
            np.copyto(self.static[self.cursor + self.history_bins], static_bin)
            np.copyto(self.dynamic[self.cursor], dynamic_bin)
            np.copyto(self.dynamic[self.cursor + self.history_bins], dynamic_bin)
            self.cursor = (self.cursor + 1) % self.history_bins
            self.filled = min(self.history_bins, self.filled + 1)
        return self.history()

    def history(self):
        start = self.cursor
        return (
            self.static[start : start + self.history_bins],
            self.dynamic[start : start + self.history_bins],
        )

    def torch_history(self, device=None):
        """Return `[1,T,2,H,W]` tensors, copying only for non-CPU devices."""
        import torch

        static, dynamic = self.history()
        static = torch.from_numpy(static).unsqueeze(0)
        dynamic = torch.from_numpy(dynamic).unsqueeze(0)
        if device is not None:
            static = static.to(device=device, non_blocking=True)
            dynamic = dynamic.to(device=device, non_blocking=True)
        return static, dynamic


class AsyncSeparatedEventVoxelizer:
    """Single-worker voxelizer for overlapping interval aggregation with I/O."""

    def __init__(self, voxelize_fn=voxelize_separated_events):
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="event-voxel"
        )
        self._voxelize_fn = voxelize_fn
        self._pending = None

    def submit(self, *args, **kwargs):
        if self._pending is not None and not self._pending.done():
            raise RuntimeError("consume the pending voxel result before submitting")
        self._pending = self._executor.submit(self._voxelize_fn, *args, **kwargs)
        return self._pending

    def result(self):
        if self._pending is None:
            raise RuntimeError("no pending voxelization")
        result = self._pending.result()
        self._pending = None
        return result

    def close(self):
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class StreamingMotionSeparator:
    """Causal, fixed-calibration motion separator for one camera stream.

    ``classify_maps`` intentionally remains the reproducible offline reference.
    This class removes work that should not occur at every online observation:

    * motion-vector convention and confidence thresholds are supplied once;
    * the pixel grid and normalized camera rays are cached;
    * only one frame interval is evaluated, without episode-sized buffers;
    * confidences stay in float32, avoiding float16/u8 round trips;
    * temporal support uses only the previous interval.

    The caller must pass the observed motion-vector frame already selected with
    the calibrated temporal offset.  With Isaac's current ``offset=1``, a step
    from frame ``k-1`` to ``k`` receives ``motion[k]``.
    """

    DEFAULT_THRESHOLDS = {
        "flow_threshold_px": 0.35,
        "relative_flow_threshold": 0.18,
        "depth_threshold_relative": 0.015,
        "photo_threshold_log": 0.04,
        "chroma_threshold": 0.015,
    }

    def __init__(self, calibration, thresholds=None):
        self.motion_gain = float(calibration["sign"]) * float(calibration["scale"])
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds is not None:
            self.thresholds.update(
                {
                    key: float(value)
                    for key, value in thresholds.items()
                    if key in self.DEFAULT_THRESHOLDS
                }
            )
        self._geometry_signature = None
        self._xx = None
        self._yy = None
        self._ray_x = None
        self._ray_y = None
        self._previous_flow_confidence = None
        self._kernel3 = np.ones((3, 3), np.uint8)

    def reset(self):
        """Clear causal temporal support while retaining camera geometry."""
        self._previous_flow_confidence = None

    def _ensure_geometry(self, depth, intrinsics):
        height, width = depth.shape
        signature = (
            height,
            width,
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        )
        if signature == self._geometry_signature:
            return
        yy, xx = np.indices((height, width), dtype=np.float32)
        fx, fy, cx, cy = signature[2:]
        self._xx = xx
        self._yy = yy
        self._ray_x = (xx - cx) / fx
        self._ray_y = (yy - cy) / fy
        float_buffers = (
            "x0",
            "y0",
            "x1",
            "y1",
            "z1",
            "temporary",
            "map_x",
            "map_y",
            "sampled_depth",
            "depth_error",
            "gray0",
            "gray1",
            "sampled_gray1",
            "light_error",
            "chroma_sum0",
            "chroma_sum1",
            "color_error",
            "relative_flow",
            "ego_magnitude",
        )
        self._scratch = {
            name: np.empty((height, width), np.float32)
            for name in float_buffers
        }
        self._flow = np.empty((height, width, 2), np.float32)
        self._chroma0 = np.empty((height, width, 3), np.float32)
        self._chroma1 = np.empty((height, width, 3), np.float32)
        self._sampled_chroma1 = np.empty((height, width, 3), np.float32)
        self._gray_u8_0 = np.empty((height, width), np.uint8)
        self._gray_u8_1 = np.empty((height, width), np.uint8)
        self._valid = np.empty((height, width), bool)
        self._geometry_signature = signature
        self._previous_flow_confidence = None

    @staticmethod
    def _gray_log_into(rgb, gray_u8, output):
        cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY, dst=gray_u8)
        np.copyto(output, gray_u8, casting="unsafe")
        np.log1p(output, out=output)
        return output

    @staticmethod
    def _chromaticity_into(rgb, output, channel_sum):
        np.copyto(output, rgb, casting="unsafe")
        output += 1.0
        np.add(output[..., 0], output[..., 1], out=channel_sum)
        np.add(channel_sum, output[..., 2], out=channel_sum)
        np.maximum(channel_sum, 1e-6, out=channel_sum)
        np.divide(output[..., 0], channel_sum, out=output[..., 0])
        np.divide(output[..., 1], channel_sum, out=output[..., 1])
        np.divide(output[..., 2], channel_sum, out=output[..., 2])
        return output

    def _ego_geometry(self, depth, pose0, pose1, intrinsics):
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        self._ensure_geometry(depth, intrinsics)
        fx, fy, cx, cy = self._geometry_signature[2:]

        # Combine both camera/world transforms first.  The offline reference
        # materializes camera points, world points, and next-camera points;
        # the direct relative transform avoids two large HxWx3 arrays.
        r0 = _rotation_wxyz(pose0[3:])
        r1 = _rotation_wxyz(pose1[3:])
        relative_rotation = r0.T @ r1
        relative_translation = (pose0[:3] - pose1[:3]) @ r1

        scratch = self._scratch
        x0, y0 = scratch["x0"], scratch["y0"]
        x1, y1, z1 = scratch["x1"], scratch["y1"], scratch["z1"]
        temporary = scratch["temporary"]
        np.multiply(self._ray_x, depth, out=x0)
        np.multiply(self._ray_y, depth, out=y0)

        def transform_component(output, column, translation):
            np.multiply(x0, float(relative_rotation[0, column]), out=output)
            np.multiply(y0, float(relative_rotation[1, column]), out=temporary)
            np.add(output, temporary, out=output)
            np.multiply(depth, float(relative_rotation[2, column]), out=temporary)
            np.add(output, temporary, out=output)
            output += float(translation)

        transform_component(x1, 0, relative_translation[0])
        transform_component(y1, 1, relative_translation[1])
        transform_component(z1, 2, relative_translation[2])
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(x1, z1, out=self._flow[..., 0])
            np.divide(y1, z1, out=self._flow[..., 1])
        self._flow[..., 0] *= fx
        self._flow[..., 1] *= fy
        self._flow[..., 0] += cx
        self._flow[..., 1] += cy
        np.copyto(scratch["map_x"], self._flow[..., 0])
        np.copyto(scratch["map_y"], self._flow[..., 1])
        self._flow[..., 0] -= self._xx
        self._flow[..., 1] -= self._yy

        valid = self._valid
        np.isfinite(depth, out=valid)
        valid &= np.isfinite(self._flow[..., 0])
        valid &= np.isfinite(self._flow[..., 1])
        valid &= np.isfinite(z1)
        valid &= depth > 0.02
        valid &= depth < 100.0
        valid &= z1 > 0.02
        valid &= scratch["map_x"] >= -1e-4
        valid &= scratch["map_x"] <= depth.shape[1] - 1 + 1e-4
        valid &= scratch["map_y"] >= -1e-4
        valid &= scratch["map_y"] <= depth.shape[0] - 1 + 1e-4
        flow = self._flow
        flow[~valid] = 0.0
        z1[~valid] = 0.0
        return flow, z1, valid

    def step(
        self,
        rgb0,
        depth0,
        pose0,
        rgb1,
        depth1,
        pose1,
        intrinsics,
        observed_motion,
    ):
        """Separate the raw-event interval between two observations.

        Returns float32 confidence maps named ``q_static``, ``q_dynamic``,
        ``q_illumination``, ``q_unknown``, and ``q_motion``.  These maps can be
        sampled at raw-event coordinates before updating an event voxel ring
        buffer.
        """
        depth0 = np.asarray(depth0, dtype=np.float32).squeeze()
        depth1 = np.asarray(depth1, dtype=np.float32).squeeze()
        observed_motion = np.asarray(observed_motion, dtype=np.float32)
        ego, predicted_depth, valid = self._ego_geometry(
            depth0, pose0, pose1, intrinsics
        )
        scratch = self._scratch
        map_x, map_y = scratch["map_x"], scratch["map_y"]

        # x0/y0 are no longer needed after projection, so reuse them for the
        # observed-flow residual instead of allocating two more HxW arrays.
        residual_x, residual_y = scratch["x0"], scratch["y0"]
        flow_error = scratch["x1"]
        np.multiply(observed_motion[..., 0], self.motion_gain, out=residual_x)
        np.subtract(residual_x, ego[..., 0], out=residual_x)
        np.multiply(observed_motion[..., 1], self.motion_gain, out=residual_y)
        np.subtract(residual_y, ego[..., 1], out=residual_y)
        np.hypot(residual_x, residual_y, out=flow_error)
        sampled_depth = cv2.remap(
            depth1,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            dst=scratch["sampled_depth"],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        relative_depth_error = scratch["depth_error"]
        np.subtract(sampled_depth, predicted_depth, out=relative_depth_error)
        np.abs(relative_depth_error, out=relative_depth_error)
        np.maximum(predicted_depth, 0.05, out=scratch["temporary"])
        np.divide(relative_depth_error, scratch["temporary"], out=relative_depth_error)

        log0 = self._gray_log_into(rgb0, self._gray_u8_0, scratch["gray0"])
        log1 = self._gray_log_into(rgb1, self._gray_u8_1, scratch["gray1"])
        sampled_log1 = cv2.remap(
            log1,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            dst=scratch["sampled_gray1"],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        log_delta = sampled_log1
        np.subtract(log_delta, log0, out=log_delta)

        # Compute the depth discontinuity once.  Radius-two edges are one
        # additional 3x3 dilation of the already radius-one mask.
        edge1 = depth_edge_mask(depth0, radius=1)
        edge2 = cv2.dilate(edge1.astype(np.uint8), self._kernel3).astype(bool)
        base = (
            valid
            & np.isfinite(relative_depth_error)
            & (relative_depth_error < 0.03)
            & ~edge1
        )
        exposure_shift = float(np.median(log_delta[base])) if np.any(base) else 0.0
        light_error = scratch["light_error"]
        np.subtract(log_delta, exposure_shift, out=light_error)
        np.abs(light_error, out=light_error)

        chroma0 = self._chromaticity_into(
            rgb0, self._chroma0, scratch["chroma_sum0"]
        )
        chroma1 = self._chromaticity_into(
            rgb1, self._chroma1, scratch["chroma_sum1"]
        )
        sampled_chroma1 = cv2.remap(
            chroma1,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            dst=self._sampled_chroma1,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        np.subtract(sampled_chroma1, chroma0, out=sampled_chroma1)
        color_error = scratch["color_error"]
        np.square(sampled_chroma1[..., 0], out=color_error)
        np.square(sampled_chroma1[..., 1], out=scratch["temporary"])
        np.add(color_error, scratch["temporary"], out=color_error)
        np.square(sampled_chroma1[..., 2], out=scratch["temporary"])
        np.add(color_error, scratch["temporary"], out=color_error)
        np.sqrt(color_error, out=color_error)

        valid &= (
            np.isfinite(flow_error)
            & np.isfinite(relative_depth_error)
            & np.isfinite(light_error)
            & np.isfinite(color_error)
        )
        reliable = valid & ~edge2
        flow_error[~valid] = 0.0
        relative_depth_error[~valid] = 0.0
        light_error[~valid] = 0.0
        color_error[~valid] = 0.0

        flow_threshold = self.thresholds["flow_threshold_px"]
        relative_flow_threshold = self.thresholds["relative_flow_threshold"]
        depth_threshold = self.thresholds["depth_threshold_relative"]
        photo_threshold = self.thresholds["photo_threshold_log"]
        chroma_threshold = self.thresholds["chroma_threshold"]

        ego_magnitude = scratch["ego_magnitude"]
        np.hypot(ego[..., 0], ego[..., 1], out=ego_magnitude)
        ego_magnitude += 1.0
        relative_flow_residual = scratch["relative_flow"]
        np.divide(flow_error, ego_magnitude, out=relative_flow_residual)
        absolute_flow_confidence = _sigmoid(
            (flow_error - flow_threshold) / max(0.12, 0.25 * flow_threshold)
        ) * reliable
        relative_flow_confidence = _sigmoid(
            (relative_flow_residual - relative_flow_threshold)
            / max(0.04, 0.25 * relative_flow_threshold)
        ) * reliable
        flow_confidence = np.maximum(
            absolute_flow_confidence, relative_flow_confidence
        )
        depth_confidence = _sigmoid(
            (relative_depth_error - depth_threshold)
            / max(0.006, 0.35 * depth_threshold)
        ) * valid
        photo_confidence = _sigmoid(
            (light_error - photo_threshold) / max(0.02, 0.35 * photo_threshold)
        ) * valid
        chroma_confidence = _sigmoid(
            (color_error - chroma_threshold)
            / max(0.008, 0.35 * chroma_threshold)
        ) * valid

        spread = _dilate_float(flow_confidence, radius=2)
        on_edge = valid & ~reliable
        flow_confidence[on_edge] = spread[on_edge]
        depth_confidence = _dilate_float(depth_confidence, radius=2) * valid
        chroma_confidence = _dilate_float(chroma_confidence, radius=1) * valid

        if (
            self._previous_flow_confidence is None
            or self._previous_flow_confidence.shape != flow_confidence.shape
        ):
            neighbor = flow_confidence
        else:
            neighbor = _dilate_float(self._previous_flow_confidence, radius=3)
        persistent = np.sqrt(np.clip(flow_confidence * neighbor, 0.0, 1.0))
        if self._previous_flow_confidence is None:
            self._previous_flow_confidence = np.empty_like(flow_confidence)
        np.copyto(self._previous_flow_confidence, flow_confidence)

        static, dynamic, illumination, unknown = fuse_motion_lighting(
            flow_confidence,
            depth_confidence,
            photo_confidence,
            chroma_confidence,
            persistent,
            valid,
        )
        return {
            "q_motion": np.clip(flow_confidence, 0.0, 1.0).astype(
                np.float32, copy=False
            ),
            "q_static": np.clip(static, 0.0, 1.0).astype(np.float32, copy=False),
            "q_dynamic": np.clip(dynamic, 0.0, 1.0).astype(
                np.float32, copy=False
            ),
            "q_illumination": np.clip(illumination, 0.0, 1.0).astype(
                np.float32, copy=False
            ),
            "q_unknown": np.clip(unknown, 0.0, 1.0).astype(
                np.float32, copy=False
            ),
            "valid": valid,
        }


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
