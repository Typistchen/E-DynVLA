"""DOM/EDV event data adapters used by E-DynVLA.

The adapters align event windows to RGB observations and expose two streams to
the policy: static-world-consistent events and independently dynamic events.
Raw event files are kept raw on disk; when confidence fields are missing, a
motion separator can be supplied to create the two streams at read time.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from policies.edynvla.motion_separation import RawEventMotionSeparator


STATIC_EVENT_KEY = "observation.events.static"
DYNAMIC_EVENT_KEY = "observation.events.dynamic"
FUTURE_EVENT_KEY = "observation.events.future_activity"
FUTURE_RGB_KEY = "observation.wam.future_rgb"
FUTURE_RGB_VALID_KEY = "observation.wam.future_rgb_valid"


@dataclass(frozen=True)
class EventWindowConfig:
    sensor: str = "wrist_cam"
    fps: float = 25.0
    history_bins: int = 8
    bin_ms: float = 10.0
    output_size: tuple[int, int] = (96, 128)
    source_size: tuple[int, int] = (360, 480)
    clip_count: float = 8.0
    future_steps: int = 10
    future_grid_size: tuple[int, int] = (12, 16)
    event_code_root: str | None = None
    # Dynamic-only training: skip the static stream and all future-activity
    # computation at read time.
    dynamic_only: bool = False

    @property
    def bin_seconds(self) -> float:
        return self.bin_ms / 1000.0


def future_frame_interpolation(
    frame_index: int,
    *,
    future_steps: int,
    bin_seconds: float,
    fps: float,
    n_frames: int,
) -> tuple[int, int, float, bool]:
    """Two RGB frame indices and blend weight at the exact WAM horizon."""
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    position = frame_index + future_steps * bin_seconds * fps
    left = int(np.floor(position))
    right = int(np.ceil(position))
    valid = right < n_frames
    left = min(max(left, 0), n_frames - 1)
    right = min(max(right, 0), n_frames - 1)
    alpha = float(position - np.floor(position)) if valid else 0.0
    return left, right, alpha, valid


def _event_window_indices(
    *,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    polarity: np.ndarray,
    start_time: float,
    num_bins: int,
    bin_seconds: float,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Flat voxel indices and validity mask for one event window."""
    out_h, out_w = output_size
    src_h, src_w = source_size
    time_bin = np.floor(
        (t.astype(np.float64) - start_time) / bin_seconds
    ).astype(np.int32)
    xx = (x.astype(np.int32) * out_w) // src_w
    yy = (y.astype(np.int32) * out_h) // src_h
    pp = (polarity > 0).astype(np.int32)
    # Unsigned views make the negative-value checks branch-free: negative
    # bins/coordinates wrap to huge uint32 values and fail the upper bound.
    valid = (
        (time_bin.view(np.uint32) < np.uint32(num_bins))
        & (xx.view(np.uint32) < np.uint32(out_w))
        & (yy.view(np.uint32) < np.uint32(out_h))
    )
    flat = ((time_bin * 2 + pp) * out_h + yy) * out_w + xx
    return flat, valid


def _accumulate_voxels(
    flat: np.ndarray,
    valid: np.ndarray,
    weight: np.ndarray,
    *,
    num_bins: int,
    output_size: tuple[int, int],
    clip_count: float,
) -> np.ndarray:
    out_h, out_w = output_size
    mask = valid & np.isfinite(weight)
    voxels = np.bincount(
        flat[mask],
        weights=np.asarray(weight, dtype=np.float64)[mask],
        minlength=num_bins * 2 * out_h * out_w,
    ).astype(np.float32)
    if clip_count > 0:
        voxels = np.log1p(np.minimum(voxels, clip_count)) / np.log1p(clip_count)
    return voxels.reshape(num_bins, 2, out_h, out_w)


def voxelize_weighted_events(
    *,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    polarity: np.ndarray,
    weight: np.ndarray,
    start_time: float,
    num_bins: int,
    bin_seconds: float,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    clip_count: float = 8.0,
) -> torch.Tensor:
    """Convert confidence-weighted events into ``[T,2,H,W]`` log voxels.

    Uses integer index math and ``np.bincount`` accumulation, which is
    several times faster than the equivalent ``np.add.at`` for the
    million-event windows produced by high-rate event cameras.
    """
    if len(t) == 0:
        return torch.zeros((num_bins, 2, *output_size), dtype=torch.float32)
    flat, valid = _event_window_indices(
        x=x,
        y=y,
        t=t,
        polarity=polarity,
        start_time=start_time,
        num_bins=num_bins,
        bin_seconds=bin_seconds,
        source_size=source_size,
        output_size=output_size,
    )
    voxels = _accumulate_voxels(
        flat,
        valid,
        weight,
        num_bins=num_bins,
        output_size=output_size,
        clip_count=clip_count,
    )
    return torch.from_numpy(voxels)


def voxelize_weighted_event_pair(
    *,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    polarity: np.ndarray,
    weight_static: np.ndarray,
    weight_dynamic: np.ndarray,
    start_time: float,
    num_bins: int,
    bin_seconds: float,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    clip_count: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Voxelize two weightings of the same event window with shared indexing."""
    if len(t) == 0:
        zeros = torch.zeros((num_bins, 2, *output_size), dtype=torch.float32)
        return zeros, zeros.clone()
    flat, valid = _event_window_indices(
        x=x,
        y=y,
        t=t,
        polarity=polarity,
        start_time=start_time,
        num_bins=num_bins,
        bin_seconds=bin_seconds,
        source_size=source_size,
        output_size=output_size,
    )
    static = _accumulate_voxels(
        flat,
        valid,
        weight_static,
        num_bins=num_bins,
        output_size=output_size,
        clip_count=clip_count,
    )
    dynamic = _accumulate_voxels(
        flat,
        valid,
        weight_dynamic,
        num_bins=num_bins,
        output_size=output_size,
        clip_count=clip_count,
    )
    return torch.from_numpy(static), torch.from_numpy(dynamic)


class SeparatedEventWindowReader:
    """Lazy aligned reader for stored-q or raw HDF5 event episodes."""

    def __init__(
        self,
        filename: str | Path,
        config: EventWindowConfig,
        *,
        support_h5: str | Path | None = None,
        rgb_frames: np.ndarray | None = None,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError("h5py is required to read separated event files") from exc

        self.filename = Path(filename)
        self.config = config
        self._file = h5py.File(self.filename, "r", libver="latest")
        self._group = self._file[f"DVS/{config.sensor}"]
        self._support_file = (
            h5py.File(support_h5, "r", libver="latest")
            if support_h5 is not None
            else None
        )
        self._rgb_frames = rgb_frames
        # Locating an event window only needs a searchsorted on timestamps.
        # Caches written by EDVSupportDataset carry a coarse ``t_index`` (one
        # entry per TIME_INDEX_STEP events) so opening a reader does not load
        # the full, possibly 80M-entry, timestamp array. Legacy files without
        # the index fall back to the full in-memory array.
        self._n_events = int(self._group["t"].shape[0])
        if "t_index" in self._group:
            self._t_index = np.asarray(self._group["t_index"])
            self._t_index_step = int(self._group.attrs.get("t_index_step", 1))
            self._timestamps = None
        else:
            self._t_index = None
            self._t_index_step = 0
            # The on-disk dtype is preserved (float64 for stored DOM H5s,
            # float32 for converted AEDAT4 caches) to halve memory for large
            # event counts.
            self._timestamps = np.asarray(self._group["t"])
        self.time_origin = float(
            self._file.attrs.get(
                "event_time_origin_s",
                self._group["t"][0] if self._n_events else 0.0,
            )
        )
        self._has_stored_confidence = all(
            key in self._group for key in ("q_static", "q_dynamic", "q_illumination")
        )
        self._motion_separator = None
        if not self._has_stored_confidence:
            if self._support_file is None or self._rgb_frames is None:
                raise RuntimeError(
                    f"{self.filename} stores raw events only. Provide support_h5 "
                    "and rgb_frames to derive static/dynamic event inputs."
                )
            self._motion_separator = RawEventMotionSeparator(
                support_h5=self._support_file,
                event_code_root=config.event_code_root,
                source_size=config.source_size,
                output_size=config.output_size,
                future_output_size=config.future_grid_size,
            )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._support_file is not None:
            self._support_file.close()
            self._support_file = None

    def __enter__(self) -> "SeparatedEventWindowReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _window_bounds(self, start_time: float, end_time: float) -> tuple[int, int]:
        """Event index range [lo, hi) covering ``[start_time, end_time)``."""
        if self._t_index is None:
            return (
                int(np.searchsorted(self._timestamps, start_time, side="left")),
                int(np.searchsorted(self._timestamps, end_time, side="left")),
            )
        step = self._t_index_step
        coarse_lo = int(np.searchsorted(self._t_index, start_time, side="left"))
        coarse_hi = int(np.searchsorted(self._t_index, end_time, side="right"))
        lo = max(0, (coarse_lo - 1) * step)
        hi = min(self._n_events, (coarse_hi + 1) * step)
        chunk = np.asarray(self._group["t"][lo:hi])
        return (
            lo + int(np.searchsorted(chunk, start_time, side="left")),
            lo + int(np.searchsorted(chunk, end_time, side="left")),
        )

    def frame(self, frame_index: int) -> dict[str, torch.Tensor]:
        """Read history ending at the timestamp of a DOM frame."""
        end_time = self.time_origin + frame_index / self.config.fps
        start_time = end_time - self.config.history_bins * self.config.bin_seconds
        lo, hi = self._window_bounds(start_time, end_time)
        arrays = self._read_slice(int(lo), int(hi))
        if self.config.dynamic_only:
            if not self._has_stored_confidence:
                _, dynamic = self._motion_separator.voxelize(
                    x=arrays["x"],
                    y=arrays["y"],
                    t=arrays["t"],
                    p=arrays["p"],
                    frame_index=frame_index,
                    rgb_frames=self._rgb_frames,
                    start_time=start_time,
                    num_bins=self.config.history_bins,
                    bin_seconds=self.config.bin_seconds,
                    clip_count=self.config.clip_count,
                )
                return {DYNAMIC_EVENT_KEY: torch.from_numpy(dynamic)}
            illumination_keep = np.clip(1.0 - arrays["q_illumination"], 0.0, 1.0)
            dynamic = voxelize_weighted_events(
                x=arrays["x"],
                y=arrays["y"],
                t=arrays["t"],
                polarity=arrays["p"],
                weight=arrays["q_dynamic"] * illumination_keep,
                start_time=start_time,
                num_bins=self.config.history_bins,
                bin_seconds=self.config.bin_seconds,
                source_size=self.config.source_size,
                output_size=self.config.output_size,
                clip_count=self.config.clip_count,
            )
            return {DYNAMIC_EVENT_KEY: dynamic}
        if not self._has_stored_confidence:
            static, dynamic = self._motion_separator.voxelize(
                x=arrays["x"],
                y=arrays["y"],
                t=arrays["t"],
                p=arrays["p"],
                frame_index=frame_index,
                rgb_frames=self._rgb_frames,
                start_time=start_time,
                num_bins=self.config.history_bins,
                bin_seconds=self.config.bin_seconds,
                clip_count=self.config.clip_count,
            )
            return {
                STATIC_EVENT_KEY: torch.from_numpy(static),
                DYNAMIC_EVENT_KEY: torch.from_numpy(dynamic),
                FUTURE_EVENT_KEY: self._future_activity(frame_index, end_time),
            }
        illumination_keep = np.clip(1.0 - arrays["q_illumination"], 0.0, 1.0)
        static, dynamic = voxelize_weighted_event_pair(
            x=arrays["x"],
            y=arrays["y"],
            t=arrays["t"],
            polarity=arrays["p"],
            weight_static=arrays["q_static"] * illumination_keep,
            weight_dynamic=arrays["q_dynamic"] * illumination_keep,
            start_time=start_time,
            num_bins=self.config.history_bins,
            bin_seconds=self.config.bin_seconds,
            source_size=self.config.source_size,
            output_size=self.config.output_size,
            clip_count=self.config.clip_count,
        )
        return {
            STATIC_EVENT_KEY: static,
            DYNAMIC_EVENT_KEY: dynamic,
            FUTURE_EVENT_KEY: self._future_activity(frame_index, end_time),
        }

    def _future_activity(self, frame_index: int, start_time: float) -> torch.Tensor:
        end_time = start_time + self.config.future_steps * self.config.bin_seconds
        lo, hi = self._window_bounds(start_time, end_time)
        arrays = self._read_slice(int(lo), int(hi))
        if not self._has_stored_confidence:
            map_index = min(frame_index + 1, len(self._rgb_frames) - 1)
            static, dynamic = self._motion_separator.voxelize(
                x=arrays["x"],
                y=arrays["y"],
                t=arrays["t"],
                p=arrays["p"],
                frame_index=map_index,
                rgb_frames=self._rgb_frames,
                start_time=start_time,
                num_bins=self.config.future_steps,
                bin_seconds=self.config.bin_seconds,
                clip_count=1.0,
                output_size=self.config.future_grid_size,
            )
            return torch.from_numpy(
                np.concatenate([static, dynamic], axis=1)
            ).gt(0).float()
        illumination_keep = np.clip(1.0 - arrays["q_illumination"], 0.0, 1.0)
        static, dynamic = voxelize_weighted_event_pair(
            x=arrays["x"],
            y=arrays["y"],
            t=arrays["t"],
            polarity=arrays["p"],
            weight_static=arrays["q_static"] * illumination_keep,
            weight_dynamic=arrays["q_dynamic"] * illumination_keep,
            start_time=start_time,
            num_bins=self.config.future_steps,
            bin_seconds=self.config.bin_seconds,
            source_size=self.config.source_size,
            output_size=self.config.future_grid_size,
            clip_count=1.0,
        )
        return torch.cat([static, dynamic], dim=1).gt(0).float()

    def _read_slice(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        keys = ["x", "y", "t", "p"]
        if self._has_stored_confidence:
            keys.extend(("q_static", "q_dynamic", "q_illumination"))
        return {key: np.asarray(self._group[key][lo:hi]) for key in keys}


class DOMEventDataset(torch.utils.data.Dataset):
    """Direct training/debug adapter for paired DOM and eventized episodes.

    This adapter intentionally returns the same RGB/state/action names used by
    DynamicVLA plus the three E-DynVLA event keys. A production conversion to
    LeRobot can reuse the same per-frame contract.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        dataset_root: str | Path,
        event_config: EventWindowConfig | None = None,
        cameras: tuple[str, ...] = ("opst_cam", "wrist_cam"),
        action_horizon: int = 20,
        max_open_event_files: int = 2,
        split: str | None = None,
        delta_action: bool = True,
        image_transforms=None,
        test_every: int = 10,
        rotation_format: str = "euler",
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest)
        self.dataset_root = Path(dataset_root)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        episodes = self.manifest["episodes"]
        if split is not None:
            if split not in ("train", "test"):
                raise ValueError(f"unknown split: {split}")
            selected = []
            for index, episode in enumerate(episodes):
                is_test = (index + 1) % test_every == 0
                if (split == "test" and is_test) or (split == "train" and not is_test):
                    selected.append(episode)
            episodes = selected
        self.episodes = episodes
        self.event_config = event_config or EventWindowConfig(
            sensor=self.manifest.get("sensor", "wrist_cam"),
            fps=float(self.manifest.get("fps", 25.0)),
        )
        self.cameras = tuple(cameras)
        self.action_horizon = action_horizon
        self.delta_action = delta_action
        self.image_transforms = image_transforms
        if rotation_format not in ("euler", "quat"):
            raise ValueError("rotation_format must be 'euler' or 'quat'")
        self.rotation_format = rotation_format
        self.max_open_event_files = max_open_event_files
        self._ends = []
        total = 0
        for episode in self.episodes:
            total += int(episode["frame_count"])
            self._ends.append(total)
        self._event_readers: OrderedDict[str, SeparatedEventWindowReader] = (
            OrderedDict()
        )
        self.camera_keys = [f"observation.images.{camera}" for camera in self.cameras]
        state_dim = 6 if rotation_format == "euler" else 7
        action_dim = state_dim + 1
        self.policy_features = {
            **{
                key: {"type": "VISUAL", "shape": (3, 360, 480)}
                for key in self.camera_keys
            },
            "observation.state": {"type": "STATE", "shape": (state_dim,)},
            "action": {"type": "ACTION", "shape": (action_dim,)},
        }
        self.stats = self._compute_stats()
        self.meta = SimpleNamespace(
            policy_features=self.policy_features,
            stats=self.stats,
            camera_keys=self.camera_keys,
            total_episodes=len(self.episodes),
            total_frames=len(self),
        )

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._ends, index)
        start = 0 if episode_index == 0 else self._ends[episode_index - 1]
        frame_index = index - start
        episode = self.episodes[episode_index]

        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError("h5py is required to read DOM episodes") from exc

        sample: dict[str, torch.Tensor | str] = {}
        dom_path = self.dataset_root / episode["dom_h5"]
        with h5py.File(dom_path, "r", libver="latest") as dom:
            for camera in self.cameras:
                rgb = np.asarray(dom[f"{camera}_rgb"][frame_index])
                sample[f"observation.images.{camera}"] = (
                    torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
                )
            state = self._state_array(dom, slice(frame_index, frame_index + 1))[0]
            sample["observation.state"] = torch.from_numpy(state)
            n_frames = int(dom["action"].shape[0])
            future_left, future_right, future_alpha, future_valid = (
                future_frame_interpolation(
                    frame_index,
                    future_steps=self.event_config.future_steps,
                    bin_seconds=self.event_config.bin_seconds,
                    fps=self.event_config.fps,
                    n_frames=n_frames,
                )
            )
            future_rgb = np.asarray(
                dom[f"{self.event_config.sensor}_rgb"][future_left], dtype=np.float32
            ).copy()
            if future_right != future_left:
                right_rgb = np.asarray(
                    dom[f"{self.event_config.sensor}_rgb"][future_right],
                    dtype=np.float32,
                )
                future_rgb = (
                    (1.0 - future_alpha) * future_rgb
                    + future_alpha * right_rgb
                )
            future_rgb = (
                torch.from_numpy(future_rgb).permute(2, 0, 1).float() / 255.0
            )
            sample[FUTURE_RGB_KEY] = future_rgb
            sample[FUTURE_RGB_VALID_KEY] = torch.tensor(future_valid)
            valid_actions = self._action_array(np.asarray(
                dom["action"][frame_index : frame_index + self.action_horizon],
                dtype=np.float32,
            ))
            valid_count = len(valid_actions)
            if valid_count < self.action_horizon:
                padding = np.repeat(
                    valid_actions[-1:], self.action_horizon - valid_count, axis=0
                )
                valid_actions = np.concatenate([valid_actions, padding], axis=0)
            sample["action"] = torch.from_numpy(valid_actions)
            action_padding = np.arange(self.action_horizon) >= valid_count
            sample["actions_id_pad"] = torch.from_numpy(action_padding)

        event_path = str((self.dataset_root / episode["event_h5"]).resolve())
        reader = self._get_event_reader(event_path)
        sample.update(reader.frame(frame_index))
        sample["task"] = self._instruction_text(episode.get("instruction", {}))
        sample["episode_index"] = torch.tensor(episode_index)
        sample["frame_index"] = torch.tensor(frame_index)
        if self.delta_action:
            action_dim = sample["action"].shape[-1] - 1
            sample["action"][..., :action_dim] -= sample["observation.state"][:action_dim]
        if self.image_transforms is not None:
            sample = self.image_transforms(
                sample, [*self.camera_keys, FUTURE_RGB_KEY]
            )
        sample[FUTURE_RGB_KEY] = F.interpolate(
            sample[FUTURE_RGB_KEY][None],
            size=self.event_config.future_grid_size,
            mode="area",
        )[0]
        return sample

    def close(self) -> None:
        for reader in self._event_readers.values():
            reader.close()
        self._event_readers.clear()

    def _get_event_reader(self, path: str) -> SeparatedEventWindowReader:
        if path in self._event_readers:
            self._event_readers.move_to_end(path)
            return self._event_readers[path]
        reader = SeparatedEventWindowReader(path, self.event_config)
        self._event_readers[path] = reader
        while len(self._event_readers) > self.max_open_event_files:
            _, old_reader = self._event_readers.popitem(last=False)
            old_reader.close()
        return reader

    def _compute_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError("h5py is required to compute DOM statistics") from exc

        states = []
        actions = []
        for episode in self.episodes:
            with h5py.File(
                self.dataset_root / episode["dom_h5"], "r", libver="latest"
            ) as dom:
                states.append(self._state_array(dom, slice(None)))
                actions.append(
                    self._action_array(np.asarray(dom["action"], dtype=np.float32))
                )
        state_array = np.concatenate(states, axis=0)
        action_array = np.concatenate(actions, axis=0)
        if self.delta_action:
            action_array = action_array.copy()
            action_array[:, :-1] -= state_array

        def stats(array: np.ndarray) -> dict[str, torch.Tensor]:
            return {
                "mean": torch.from_numpy(array.mean(axis=0).astype(np.float32)),
                "std": torch.from_numpy(array.std(axis=0).astype(np.float32)).clamp_min(1e-6),
                "min": torch.from_numpy(array.min(axis=0).astype(np.float32)),
                "max": torch.from_numpy(array.max(axis=0).astype(np.float32)),
                "count": torch.tensor(array.shape[0]),
            }

        return {
            "observation.state": stats(state_array),
            "action": stats(action_array),
        }

    def _state_array(self, dom, selection) -> np.ndarray:
        position = np.asarray(dom["ee_pos"][selection], dtype=np.float32)
        quaternion = np.asarray(dom["ee_quat"][selection], dtype=np.float32)
        rotation = self._rotation_array(quaternion)
        return np.concatenate([position, rotation], axis=-1)

    def _action_array(self, action: np.ndarray) -> np.ndarray:
        if self.rotation_format == "quat":
            return action
        rotation = self._rotation_array(action[..., 3:7])
        return np.concatenate([action[..., :3], rotation, action[..., -1:]], axis=-1)

    def _rotation_array(self, quaternion: np.ndarray) -> np.ndarray:
        if self.rotation_format == "quat":
            return quaternion
        # Isaac/DOM stores scalar-first quaternions (w, x, y, z).
        w, x, y, z = np.moveaxis(quaternion, -1, 0)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        euler = np.stack([roll, pitch, yaw], axis=-1)
        # Match DynamicVLA's continuous convention for roll and yaw.
        euler[..., [0, 2]] = np.mod(euler[..., [0, 2]], 2 * np.pi)
        return euler.astype(np.float32)

    @staticmethod
    def _instruction_text(instruction: dict) -> str:
        task = instruction.get("task", "pick")
        objects = instruction.get("objects") or ["object"]
        containers = instruction.get("containers") or ["container"]
        if task == "place":
            return f"Place the {objects[0]} in the {containers[0]}."
        return f"Pick up the {objects[0]}."
