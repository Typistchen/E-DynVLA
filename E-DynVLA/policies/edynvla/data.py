"""DOM + separated-event data adapter used by E-DynVLA.

The adapter reads the HDF5 files produced by
``separate_dynamic_static_events.py`` and aligns event windows to DOM frames.
It keeps simulator-only segmentation and object velocity out of model inputs.
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

    @property
    def bin_seconds(self) -> float:
        return self.bin_ms / 1000.0


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
    """Convert confidence-weighted events into ``[T,2,H,W]`` log voxels."""
    out_h, out_w = output_size
    src_h, src_w = source_size
    voxels = np.zeros((num_bins, 2, out_h, out_w), dtype=np.float32)
    if len(t) == 0:
        return torch.from_numpy(voxels)

    time_bin = np.floor((t - start_time) / bin_seconds).astype(np.int64)
    xx = np.floor(x.astype(np.float64) * out_w / src_w).astype(np.int64)
    yy = np.floor(y.astype(np.float64) * out_h / src_h).astype(np.int64)
    pp = (polarity > 0).astype(np.int64)
    valid = (
        (time_bin >= 0)
        & (time_bin < num_bins)
        & (xx >= 0)
        & (xx < out_w)
        & (yy >= 0)
        & (yy < out_h)
        & np.isfinite(weight)
    )
    np.add.at(
        voxels,
        (time_bin[valid], pp[valid], yy[valid], xx[valid]),
        weight[valid].astype(np.float32),
    )
    if clip_count > 0:
        voxels = np.log1p(np.minimum(voxels, clip_count)) / np.log1p(clip_count)
    return torch.from_numpy(voxels)


class SeparatedEventWindowReader:
    """Lazy aligned reader for one separated-event HDF5 episode."""

    def __init__(self, filename: str | Path, config: EventWindowConfig) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError("h5py is required to read separated event files") from exc

        self.filename = Path(filename)
        self.config = config
        self._file = h5py.File(self.filename, "r", libver="latest")
        self._group = self._file[f"DVS/{config.sensor}"]
        # One timestamp array per open episode keeps repeated searchsorted calls
        # fast. Dataset workers should each construct their own reader.
        self._timestamps = np.asarray(self._group["t"], dtype=np.float64)
        self.time_origin = float(
            self._file.attrs.get(
                "event_time_origin_s",
                self._timestamps[0] if len(self._timestamps) else 0.0,
            )
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "SeparatedEventWindowReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def frame(self, frame_index: int) -> dict[str, torch.Tensor]:
        """Read history ending at the timestamp of a DOM frame."""
        end_time = self.time_origin + frame_index / self.config.fps
        start_time = end_time - self.config.history_bins * self.config.bin_seconds
        lo, hi = np.searchsorted(
            self._timestamps, [start_time, end_time], side="left"
        )
        arrays = self._read_slice(int(lo), int(hi))
        illumination_keep = np.clip(1.0 - arrays["q_illumination"], 0.0, 1.0)
        common = dict(
            x=arrays["x"],
            y=arrays["y"],
            t=arrays["t"],
            polarity=arrays["p"],
            start_time=start_time,
            num_bins=self.config.history_bins,
            bin_seconds=self.config.bin_seconds,
            source_size=self.config.source_size,
            output_size=self.config.output_size,
            clip_count=self.config.clip_count,
        )
        static = voxelize_weighted_events(
            **common, weight=arrays["q_static"] * illumination_keep
        )
        dynamic = voxelize_weighted_events(
            **common, weight=arrays["q_dynamic"] * illumination_keep
        )
        return {
            STATIC_EVENT_KEY: static,
            DYNAMIC_EVENT_KEY: dynamic,
            FUTURE_EVENT_KEY: self._future_activity(end_time),
        }

    def _future_activity(self, start_time: float) -> torch.Tensor:
        end_time = start_time + self.config.future_steps * self.config.bin_seconds
        lo, hi = np.searchsorted(
            self._timestamps, [start_time, end_time], side="left"
        )
        arrays = self._read_slice(int(lo), int(hi))
        illumination_keep = np.clip(1.0 - arrays["q_illumination"], 0.0, 1.0)
        grid_size = self.config.future_grid_size
        static = voxelize_weighted_events(
            x=arrays["x"],
            y=arrays["y"],
            t=arrays["t"],
            polarity=arrays["p"],
            weight=arrays["q_static"] * illumination_keep,
            start_time=start_time,
            num_bins=self.config.future_steps,
            bin_seconds=self.config.bin_seconds,
            source_size=self.config.source_size,
            output_size=grid_size,
            clip_count=1.0,
        )
        dynamic = voxelize_weighted_events(
            x=arrays["x"],
            y=arrays["y"],
            t=arrays["t"],
            polarity=arrays["p"],
            weight=arrays["q_dynamic"] * illumination_keep,
            start_time=start_time,
            num_bins=self.config.future_steps,
            bin_seconds=self.config.bin_seconds,
            source_size=self.config.source_size,
            output_size=grid_size,
            clip_count=1.0,
        )
        return torch.cat([static, dynamic], dim=1).gt(0).float()

    def _read_slice(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        keys = ("x", "y", "t", "p", "q_static", "q_dynamic", "q_illumination")
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
            future_offset = max(
                1,
                round(
                    self.event_config.future_steps
                    * self.event_config.bin_seconds
                    * self.event_config.fps
                ),
            )
            future_index = frame_index + future_offset
            clamped_future_index = min(future_index, n_frames - 1)
            future_rgb = np.asarray(
                dom[f"{self.event_config.sensor}_rgb"][clamped_future_index]
            )
            future_rgb = (
                torch.from_numpy(future_rgb.copy()).permute(2, 0, 1).float() / 255.0
            )
            sample[FUTURE_RGB_KEY] = future_rgb
            sample[FUTURE_RGB_VALID_KEY] = torch.tensor(future_index < n_frames)
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
