import os
from pathlib import Path

import numpy as np
import pytest
import torch

from policies.edynvla.edv_support import (
    EDVSupportDataset,
    event_h5_has_confidence,
    select_edv_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EDV_ROOT = Path(os.getenv("EDV_SUPPORT_ROOT", "/home/typist/dataset/EDV_Support"))


def _sample(index, success=True, split="unsplit"):
    return {
        "sample_index": index,
        "relative_path": f"sample_{index:06d}",
        "split": split,
        "success": success,
        "frame_count": 10,
    }


def test_select_edv_samples_excludes_failures_and_splits_every_nth():
    samples = [_sample(i) for i in range(25)]
    samples[3]["success"] = False
    exists = lambda sample: True  # noqa: E731

    train = select_edv_samples(
        samples, split="train", test_every=10, exclude_failures=True, sample_exists=exists
    )
    test = select_edv_samples(
        samples, split="test", test_every=10, exclude_failures=True, sample_exists=exists
    )
    # The position counter skips the failed sample, so the first test sample
    # is the 10th surviving candidate.
    assert [s["sample_index"] for s in test] == [10, 20]
    assert 3 not in [s["sample_index"] for s in train + test]
    assert len(train) == 22


def test_select_edv_samples_honors_explicit_split_assignment():
    samples = [_sample(0, split="train"), _sample(1, split="test"), _sample(2)]
    exists = lambda sample: True  # noqa: E731
    train = select_edv_samples(
        samples, split="train", test_every=2, exclude_failures=True, sample_exists=exists
    )
    test = select_edv_samples(
        samples, split="test", test_every=2, exclude_failures=True, sample_exists=exists
    )
    # Sample 1 is explicitly test; sample 2 sits at position 2, which is not
    # a (position+1) % 2 == 0 test slot, so it stays in train.
    assert [s["sample_index"] for s in train] == [0, 2]
    assert [s["sample_index"] for s in test] == [1]


def test_select_edv_samples_skips_missing_directories():
    samples = [_sample(0), _sample(1), _sample(2)]
    exists = lambda sample: sample["sample_index"] != 1  # noqa: E731
    train = select_edv_samples(
        samples, split="train", test_every=10, exclude_failures=True, sample_exists=exists
    )
    assert [s["sample_index"] for s in train] == [0, 2]


def _missing_requirements():
    missing = []
    if not (EDV_ROOT / "dataset_info.json").is_file():
        missing.append(f"EDV dataset not found at {EDV_ROOT} (set EDV_SUPPORT_ROOT)")
    for module in ("pyarrow", "cv2", "h5py", "dv_processing", "torch"):
        try:
            __import__(module)
        except ImportError:
            missing.append(f"module {module}")
    return missing


_requires_local_edv_dataset = pytest.mark.skipif(
    len(_missing_requirements()) > 0,
    reason="; ".join(_missing_requirements()),
)


@_requires_local_edv_dataset
class TestEDVSupportDatasetIntegration:
    @pytest.fixture(scope="module")
    def dataset(self):
        missing = _missing_requirements()
        if missing:
            pytest.skip("; ".join(missing))
        return EDVSupportDataset(
            EDV_ROOT,
            split="train",
            event_code_root=REPO_ROOT / "V2E-VLA",
            observation_deltas=(-2, 0),
        )

    def test_sample_contract(self, dataset):
        assert len(dataset) > 0
        sample = dataset[0]
        expected_keys = {
            "observation.images.opst_cam",
            "observation.images.wrist_cam",
            "observation.state",
            "observation.wam.future_rgb",
            "observation.wam.future_rgb_valid",
            "action",
            "actions_id_pad",
            "observation.events.static",
            "observation.events.dynamic",
            "observation.events.future_activity",
            "task",
            "episode_index",
            "frame_index",
        }
        assert set(sample.keys()) == expected_keys
        assert sample["observation.images.opst_cam"].shape == (2, 3, 360, 480)
        assert sample["observation.images.wrist_cam"].shape == (2, 3, 360, 480)
        assert sample["observation.images.opst_cam"].min() >= 0.0
        assert sample["observation.images.opst_cam"].max() <= 1.0
        assert sample["observation.state"].shape == (6,)
        assert sample["action"].shape == (20, 7)
        assert sample["actions_id_pad"].dtype == torch.bool
        assert sample["observation.events.static"].shape == (8, 2, 96, 128)
        assert sample["observation.events.dynamic"].shape == (8, 2, 96, 128)
        assert sample["observation.events.future_activity"].shape == (10, 4, 12, 16)
        assert sample["observation.events.future_activity"].max() <= 1.0
        assert sample["observation.wam.future_rgb"].shape == (3, 12, 16)
        assert "Pick up the" in sample["task"]

    def test_same_episode_samples_share_bundle(self, dataset):
        dataset[0]
        bundles_after_first = len(dataset._bundles)
        dataset[1]
        assert len(dataset._bundles) == bundles_after_first

    def test_event_cache_stores_separation_confidences(self, dataset):
        cache_dir = dataset.event_cache_root
        sample_index = dataset.samples[0]["sample_index"]
        sensor = dataset.event_config.sensor
        expected = cache_dir / f"sample_{sample_index:06d}_{sensor}.h5"
        assert expected.is_file()
        assert event_h5_has_confidence(expected, sensor)

    def test_bundle_uses_stored_q_fast_path(self, dataset):
        sample = dataset.samples[0]
        bundle = dataset._get_sample_bundle(sample)
        # Stored-q readers keep no in-memory timestamp array and no separator.
        assert bundle.reader._timestamps is None
        assert bundle.reader._t_index is not None
        assert bundle.reader._motion_separator is None
        assert set(bundle.captures) == set(dataset.cameras)
