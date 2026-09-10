"""EDV_Support (edv-4.0) dataset adapter for E-DynVLA.

Reads the per-sample EDV layout directly (RGB mp4 + parquet + raw AEDAT4
events + motion-separation support HDF5) and returns samples with the exact
same contract as ``DOMEventDataset``, so the policy and training code are
unchanged.  Raw events are kept raw on disk; static/dynamic separation runs
at read time through ``RawEventMotionSeparator`` via
``SeparatedEventWindowReader``.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from policies.edynvla.data import (
    DYNAMIC_EVENT_KEY,
    STATIC_EVENT_KEY,
    EventWindowConfig,
    FUTURE_RGB_KEY,
    FUTURE_RGB_VALID_KEY,
    SeparatedEventWindowReader,
)


logger = logging.getLogger(__name__)


def read_aedat4_events(path: str | Path, expected_count: int | None = None) -> dict:
    """Read one AEDAT4 event file into seconds-based numpy arrays."""
    try:
        import dv_processing as dv
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "dv_processing is required to read AEDAT4 event files; install it "
            "in the training environment or pre-convert the events"
        ) from exc

    reader = dv.io.MonoCameraRecording(str(path))
    xs, ys, ts, ps = [], [], [], []

    def _accumulate(events) -> None:
        xs.append(events["x"])
        ys.append(events["y"])
        ts.append(events["timestamp"] * 1e-6)
        ps.append(events["polarity"])

    if expected_count is not None and expected_count > 0:
        x = np.empty(expected_count, np.int16)
        y = np.empty(expected_count, np.int16)
        t = np.empty(expected_count, np.float32)
        p = np.empty(expected_count, np.int8)
        cursor = 0
        while reader.isRunning():
            batch = reader.getNextEventBatch()
            if batch is None or batch.isEmpty():
                continue
            events = batch.numpy()
            count = len(events)
            if cursor + count <= expected_count:
                x[cursor : cursor + count] = events["x"]
                y[cursor : cursor + count] = events["y"]
                t[cursor : cursor + count] = events["timestamp"] * 1e-6
                p[cursor : cursor + count] = events["polarity"]
            else:
                _accumulate(events)
            cursor += count
        if cursor != expected_count or xs:
            logger.warning(
                "%s: event count mismatch (read %d, expected %d); reallocating",
                path,
                cursor,
                expected_count,
            )
            return read_aedat4_events(path, expected_count=None)
        result = {"x": x, "y": y, "t": t, "p": p}
    else:
        while reader.isRunning():
            batch = reader.getNextEventBatch()
            if batch is None or batch.isEmpty():
                continue
            _accumulate(batch.numpy())
        result = {
            "x": np.concatenate(xs) if xs else np.empty(0, np.int16),
            "y": np.concatenate(ys) if ys else np.empty(0, np.int16),
            "t": np.concatenate(ts) if ts else np.empty(0, np.float32),
            "p": np.concatenate(ps) if ps else np.empty(0, np.int8),
        }
    order = np.argsort(result["t"], kind="stable")
    return {key: value[order] for key, value in result.items()}


def ensure_event_h5(
    aedat4_path: str | Path,
    cache_path: str | Path,
    *,
    sensor: str,
    time_origin_s: float,
    expected_count: int | None = None,
) -> Path:
    """Convert one AEDAT4 file into the reader's event-H5 layout (cached)."""
    import h5py

    cache_path = Path(cache_path)
    if cache_path.is_file():
        return cache_path
    events = read_aedat4_events(aedat4_path, expected_count=expected_count)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp{os.getpid()}")
    try:
        with h5py.File(tmp_path, "w", libver="latest") as output:
            output.attrs["event_time_origin_s"] = float(time_origin_s)
            output.attrs["sensor"] = sensor
            output.attrs["source_events"] = str(Path(aedat4_path).resolve())
            group = output.create_group(f"DVS/{sensor}")
            group.create_dataset("x", data=events["x"])
            group.create_dataset("y", data=events["y"])
            group.create_dataset("t", data=events["t"])
            group.create_dataset("p", data=events["p"])
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return cache_path


def decode_video(path: str | Path) -> np.ndarray:
    """Decode a full mp4 into ``[T, H, W, 3]`` uint8 RGB frames."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("opencv-python is required to decode EDV RGB videos") from exc

    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def select_edv_samples(
    samples: list[dict],
    *,
    split: str | None,
    test_every: int = 10,
    exclude_failures: bool = True,
    sample_exists=None,
) -> list[dict]:
    """Pick the subset of dataset_info samples for one split.

    Samples with an explicit ``split`` field (train/test) are honored;
    otherwise every ``test_every``-th remaining candidate becomes a test
    sample, matching the DOMEventDataset convention.  Missing directories and
    (optionally) failed simulations are dropped with a warning.
    """
    if split is not None and split not in ("train", "test"):
        raise ValueError(f"unknown split: {split}")
    selected = []
    position = 0
    for sample in samples:
        if sample_exists is not None and not sample_exists(sample):
            logger.warning(
                "EDV sample directory missing, skipping: %s", sample.get("relative_path")
            )
            continue
        if exclude_failures and sample.get("success") is False:
            logger.warning(
                "Excluding failed EDV sample: %s", sample.get("relative_path")
            )
            continue
        assigned = sample.get("split")
        if assigned in ("train", "test"):
            take = assigned == split
        else:
            is_test = (position + 1) % test_every == 0
            take = is_test if split == "test" else not is_test
        position += 1
        if split is None or take:
            selected.append(sample)
    return selected


class _SampleBundle:
    """Open per-sample resources: event reader + non-sensor video capture."""

    def __init__(self, reader: SeparatedEventWindowReader, captures: dict) -> None:
        self.reader = reader
        self.captures = captures

    def close(self) -> None:
        self.reader.close()
        for capture in self.captures.values():
            capture.release()
        self.captures.clear()


class EDVSupportDataset(torch.utils.data.Dataset):
    """Training adapter for the EDV_Support (edv-4.0) sample layout."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        event_config: EventWindowConfig | None = None,
        cameras: tuple[str, ...] = ("opst_cam", "wrist_cam"),
        event_sensor: str = "wrist_cam",
        action_horizon: int = 20,
        delta_action: bool = True,
        image_transforms=None,
        test_every: int = 10,
        exclude_failures: bool = True,
        event_cache_root: str | Path | None = None,
        event_code_root: str | Path | None = None,
        max_open_samples: int = 2,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.info = json.loads((self.root / "dataset_info.json").read_text(encoding="utf-8"))
        self.split = split
        self.cameras = tuple(cameras)
        self.event_sensor = event_sensor
        self.action_horizon = action_horizon
        self.delta_action = delta_action
        self.image_transforms = image_transforms
        self.max_open_samples = max_open_samples

        if event_config is None:
            event_config = EventWindowConfig(
                sensor=event_sensor,
                fps=float(self.info.get("observation_fps", 25.0)),
                event_code_root=event_code_root,
            )
        self.event_config = event_config
        if self.event_config.event_code_root is None:
            self.event_config.event_code_root = event_code_root
        self.event_cache_root = Path(
            event_cache_root or self.root / "derived_cache" / "events"
        )

        exists = lambda sample: all(  # noqa: E731
            path.is_file()
            for path in self._required_sample_files(sample)
        )
        self.samples = select_edv_samples(
            self.info["samples"],
            split=split,
            test_every=test_every,
            exclude_failures=exclude_failures,
            sample_exists=exists,
        )
        if not self.samples:
            raise RuntimeError(f"No EDV samples selected from {self.root} (split={split})")

        self._ends = []
        total = 0
        for sample in self.samples:
            total += int(sample["frame_count"])
            self._ends.append(total)
        self._bundles: OrderedDict[int, _SampleBundle] = OrderedDict()
        self._parquets: dict[int, dict] = {}
        self._instructions: dict[int, str] = {}

        self.camera_keys = [f"observation.images.{camera}" for camera in self.cameras]
        state_dim, action_dim = self._measure_dims()
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
            total_episodes=len(self.samples),
            total_frames=len(self),
        )

    # ------------------------------------------------------------------ #
    # Indexing helpers

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def _locate(self, index: int) -> tuple[int, int]:
        sample_pos = bisect.bisect_right(self._ends, index)
        start = 0 if sample_pos == 0 else self._ends[sample_pos - 1]
        return sample_pos, index - start

    # ------------------------------------------------------------------ #
    # Per-sample IO

    def _sample_dir(self, sample: dict) -> Path:
        return self.root / sample["relative_path"]

    def _required_sample_files(self, sample: dict) -> list[Path]:
        """Files every sample must provide before it can be used."""
        sample_dir = self._sample_dir(sample)
        data_dir = sample_dir / "data"
        parquets = sorted(data_dir.glob("*.parquet")) if data_dir.is_dir() else []
        return [
            parquets[0] if parquets else data_dir / "episode_missing.parquet",
            sample_dir / "rgb" / f"{self.event_sensor}.mp4",
            sample_dir / "events" / f"{self.event_sensor}.aedat4",
            sample_dir / "support" / f"{self.event_sensor}_motion_support.h5",
        ]

    def _load_parquet(self, sample: dict) -> dict:
        key = sample["sample_index"]
        if key in self._parquets:
            return self._parquets[key]
        import pyarrow.parquet as pq

        parquet_path = next((self._sample_dir(sample) / "data").glob("*.parquet"))
        table = pq.read_table(parquet_path)
        data = {
            "state": np.asarray(
                table.column("observation.state").to_pylist(), dtype=np.float32
            ),
            "action": np.asarray(table.column("action").to_pylist(), dtype=np.float32),
            "timestamp": np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64),
        }
        self._parquets[key] = data
        return data

    def _instruction_text(self, sample: dict) -> str:
        key = sample["sample_index"]
        if key not in self._instructions:
            reproduction = json.loads(
                (self._sample_dir(sample) / "reproduction.json").read_text(encoding="utf-8")
            )
            category = reproduction.get("initial_condition", {}).get(
                "object_category", "object"
            )
            self._instructions[key] = f"Pick up the {category}."
        return self._instructions[key]

    def _ensure_event_h5(self, sample: dict) -> Path:
        sensor = self.event_config.sensor
        aedat4_path = self._sample_dir(sample) / "events" / f"{sensor}.aedat4"
        cache_path = self.event_cache_root / f"sample_{sample['sample_index']:06d}_{sensor}.h5"
        timestamps = self._load_parquet(sample)["timestamp"]
        return ensure_event_h5(
            aedat4_path,
            cache_path,
            sensor=sensor,
            time_origin_s=float(timestamps[0]) if len(timestamps) else 0.0,
            expected_count=sample.get("event_counts", {}).get(sensor),
        )

    def _get_sample_bundle(self, sample: dict) -> _SampleBundle:
        key = sample["sample_index"]
        if key in self._bundles:
            self._bundles.move_to_end(key)
            return self._bundles[key]

        sensor = self.event_config.sensor
        event_h5 = self._ensure_event_h5(sample)
        wrist_frames = decode_video(self._sample_dir(sample) / "rgb" / f"{sensor}.mp4")
        support_h5 = self._sample_dir(sample) / "support" / f"{sensor}_motion_support.h5"
        reader = SeparatedEventWindowReader(
            event_h5,
            self.event_config,
            support_h5=support_h5,
            rgb_frames=wrist_frames,
        )
        captures = {}
        for camera in self.cameras:
            if camera == sensor:
                continue
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover
                raise ImportError("opencv-python is required for EDV RGB videos") from exc
            captures[camera] = cv2.VideoCapture(
                str(self._sample_dir(sample) / "rgb" / f"{camera}.mp4")
            )
        bundle = _SampleBundle(reader=reader, captures=captures)
        self._bundles[key] = bundle
        while len(self._bundles) > self.max_open_samples:
            _, old_bundle = self._bundles.popitem(last=False)
            old_bundle.close()
        return bundle

    def _video_frame(self, bundle: _SampleBundle, camera: str, frame_index: int) -> np.ndarray:
        sensor = self.event_config.sensor
        if camera == sensor:
            frames = bundle.reader._rgb_frames
            return frames[min(frame_index, len(frames) - 1)].copy()
        import cv2

        capture = bundle.captures[camera]
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(
                f"Failed to decode frame {frame_index} for camera {camera}"
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------ #
    # Dataset contract

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample_pos, frame_index = self._locate(index)
        sample = self.samples[sample_pos]
        bundle = self._get_sample_bundle(sample)

        sample_data: dict[str, torch.Tensor | str] = {}
        for camera in self.cameras:
            rgb = self._video_frame(bundle, camera, frame_index)
            sample_data[f"observation.images.{camera}"] = (
                torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            )

        parquet = self._load_parquet(sample)
        sample_data["observation.state"] = torch.from_numpy(
            parquet["state"][frame_index]
        )

        n_frames = len(parquet["action"])
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
        future_rgb = self._video_frame(bundle, self.event_config.sensor, clamped_future_index)
        sample_data[FUTURE_RGB_KEY] = (
            torch.from_numpy(future_rgb).permute(2, 0, 1).float() / 255.0
        )
        sample_data[FUTURE_RGB_VALID_KEY] = torch.tensor(future_index < n_frames)

        actions = parquet["action"][frame_index : frame_index + self.action_horizon]
        valid_count = len(actions)
        if valid_count < self.action_horizon:
            padding = np.repeat(
                actions[-1:], self.action_horizon - valid_count, axis=0
            )
            actions = np.concatenate([actions, padding], axis=0)
        sample_data["action"] = torch.from_numpy(np.asarray(actions, dtype=np.float32))
        sample_data["actions_id_pad"] = torch.from_numpy(
            np.arange(self.action_horizon) >= valid_count
        )

        sample_data.update(bundle.reader.frame(frame_index))
        sample_data["task"] = self._instruction_text(sample)
        sample_data["episode_index"] = torch.tensor(sample_pos)
        sample_data["frame_index"] = torch.tensor(frame_index)

        if self.delta_action:
            action_dim = sample_data["action"].shape[-1] - 1
            sample_data["action"][..., :action_dim] -= sample_data["observation.state"][:action_dim]
        if self.image_transforms is not None:
            sample_data = self.image_transforms(
                sample_data, [*self.camera_keys, FUTURE_RGB_KEY]
            )
        sample_data[FUTURE_RGB_KEY] = F.interpolate(
            sample_data[FUTURE_RGB_KEY][None],
            size=self.event_config.future_grid_size,
            mode="area",
        )[0]
        return sample_data

    def close(self) -> None:
        for bundle in self._bundles.values():
            bundle.close()
        self._bundles.clear()

    # ------------------------------------------------------------------ #
    # Metadata

    def _measure_dims(self) -> tuple[int, int]:
        parquet = self._load_parquet(self.samples[0])
        return parquet["state"].shape[-1], parquet["action"].shape[-1]

    def _compute_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        states = []
        actions = []
        for sample in self.samples:
            parquet = self._load_parquet(sample)
            states.append(parquet["state"])
            actions.append(parquet["action"])
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(root='{self.root}', split='{self.split}', "
            f"samples={len(self.samples)}, frames={len(self)}, "
            f"cameras={self.camera_keys}, sensor='{self.event_config.sensor}')"
        )
