"""Online static/dynamic separation for raw event windows.

This module is a thin training-side wrapper around the V2E-VLA motion
separator.  It keeps raw events on disk and derives static/dynamic event
voxels only when a sample is read.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


DEFAULT_CALIBRATION = {
    "sign": 1.0,
    "scale": 1.0,
}


class MotionSeparationUnavailable(RuntimeError):
    """Raised when raw events cannot be split with the available sample data."""


def _load_reference_module(event_code_root: str | Path | None):
    if event_code_root is None:
        raise MotionSeparationUnavailable(
            "Raw-event motion separation needs event_code_root pointing to V2E-VLA."
        )
    root = Path(event_code_root)
    module_path = root / "scripts" / "separate_dynamic_static_events.py"
    if not module_path.is_file():
        raise MotionSeparationUnavailable(
            f"Cannot find motion separator module: {module_path}"
        )
    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "edynvla_reference_motion_separation", module_path
    )
    if spec is None or spec.loader is None:
        raise MotionSeparationUnavailable(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RawEventMotionSeparator:
    """Convert raw event windows into static/dynamic voxels.

    The support file must contain per-observation arrays named
    ``depth_metric``, ``motion_vectors``, ``pose_w_ros``, and ``intrinsics``.
    """

    def __init__(
        self,
        *,
        support_h5,
        event_code_root: str | Path | None,
        source_size=(360, 480),
        output_size=(96, 128),
        calibration: dict | None = None,
        thresholds: dict | None = None,
    ) -> None:
        module = _load_reference_module(event_code_root)
        self.separator = module.StreamingMotionSeparator(
            calibration or DEFAULT_CALIBRATION,
            thresholds=thresholds,
        )
        self.voxelizer = module.ReusableSeparatedEventVoxelizer(
            source_size=source_size,
            output_size=output_size,
        )
        self.support_h5 = support_h5
        self._maps_cache: dict[int, dict[str, np.ndarray]] = {}

    def confidence_maps(self, frame_index: int, rgb_frames: np.ndarray) -> dict[str, np.ndarray]:
        if frame_index in self._maps_cache:
            return self._maps_cache[frame_index]
        if frame_index <= 0:
            maps = self._neutral_maps(frame_index)
        else:
            support = self.support_h5
            maps = self.separator.step(
                rgb_frames[frame_index - 1],
                support["depth_metric"][frame_index - 1],
                support["pose_w_ros"][frame_index - 1],
                rgb_frames[frame_index],
                support["depth_metric"][frame_index],
                support["pose_w_ros"][frame_index],
                support["intrinsics"][frame_index - 1],
                support["motion_vectors"][frame_index],
            )
        self._maps_cache[frame_index] = maps
        if len(self._maps_cache) > 32:
            oldest = min(self._maps_cache)
            self._maps_cache.pop(oldest, None)
        return maps

    def voxelize(
        self,
        *,
        x,
        y,
        t,
        p,
        frame_index: int,
        rgb_frames: np.ndarray,
        start_time: float,
        num_bins: int,
        bin_seconds: float,
        clip_count: float,
    ):
        maps = self.confidence_maps(frame_index, rgb_frames)
        return self.voxelizer.voxelize(
            x,
            y,
            t,
            p,
            maps,
            start_time=start_time,
            num_bins=num_bins,
            bin_seconds=bin_seconds,
            clip_count=clip_count,
            assume_valid=False,
        )

    def _neutral_maps(self, frame_index: int) -> dict[str, np.ndarray]:
        depth = np.asarray(self.support_h5["depth_metric"][frame_index]).squeeze()
        shape = depth.shape
        ones = np.ones(shape, dtype=np.float32)
        zeros = np.zeros(shape, dtype=np.float32)
        return {
            "q_static": ones,
            "q_dynamic": zeros,
            "q_illumination": zeros,
            "q_unknown": zeros,
            "q_motion": zeros,
            "valid": ones.astype(bool),
        }
