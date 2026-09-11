import numpy as np
import torch

from policies.dynamicvla.modeling_dynamicvla import make_att_2d_masks
from policies.edynvla.data import (
    future_frame_interpolation,
    voxelize_weighted_event_pair,
    voxelize_weighted_events,
)
from policies.edynvla.event_tokenizer import SparseEventTokenizer


def test_future_rgb_interpolates_at_exact_event_horizon():
    left, right, alpha, valid = future_frame_interpolation(
        4,
        future_steps=10,
        bin_seconds=0.01,
        fps=25.0,
        n_frames=20,
    )
    assert (left, right) == (6, 7)
    assert alpha == 0.5
    assert valid


def test_future_rgb_marks_truncated_horizon_invalid():
    left, right, alpha, valid = future_frame_interpolation(
        18,
        future_steps=10,
        bin_seconds=0.01,
        fps=25.0,
        n_frames=20,
    )
    assert (left, right, alpha) == (19, 19, 0.0)
    assert not valid


def test_weighted_voxelization_preserves_time_polarity_and_weight():
    voxels = voxelize_weighted_events(
        x=np.array([0, 31, 16]),
        y=np.array([0, 31, 16]),
        t=np.array([0.001, 0.011, 0.019]),
        polarity=np.array([-1, 1, 1]),
        weight=np.array([1.0, 0.5, 0.5]),
        start_time=0.0,
        num_bins=2,
        bin_seconds=0.01,
        source_size=(32, 32),
        output_size=(16, 16),
        clip_count=0.0,
    )
    assert voxels.shape == (2, 2, 16, 16)
    assert voxels[0, 0, 0, 0] == 1.0
    assert voxels[1, 1].sum() == 1.0


def test_paired_voxelization_matches_individual_calls():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.integers(0, 480, n).astype(np.int16)
    y = rng.integers(0, 360, n).astype(np.int16)
    t = np.sort(rng.uniform(0.0, 0.08, n)).astype(np.float32)
    polarity = rng.integers(0, 2, n).astype(np.int8)
    weight_static = rng.uniform(0.0, 1.0, n).astype(np.float32)
    weight_dynamic = rng.uniform(0.0, 1.0, n).astype(np.float32)
    kwargs = dict(
        x=x,
        y=y,
        t=t,
        polarity=polarity,
        start_time=0.0,
        num_bins=8,
        bin_seconds=0.01,
        source_size=(360, 480),
        output_size=(96, 128),
        clip_count=8.0,
    )
    static, dynamic = voxelize_weighted_event_pair(
        **kwargs,
        weight_static=weight_static,
        weight_dynamic=weight_dynamic,
    )
    assert torch.equal(
        static, voxelize_weighted_events(**kwargs, weight=weight_static)
    )
    assert torch.equal(
        dynamic, voxelize_weighted_events(**kwargs, weight=weight_dynamic)
    )


def test_sparse_tokenizer_dynamic_only_masks_empty_patches():
    tokenizer = SparseEventTokenizer(
        hidden_dim=32,
        output_dim=48,
        patch_size=8,
        max_patches_per_bin=2,
        history_bins=3,
        num_layers=1,
        num_heads=4,
    )
    dynamic = torch.zeros(2, 3, 2, 16, 16)
    dynamic[:, 0, 0, :8, :8] = 1
    dynamic[:, 2, 1, 8:, 8:] = 1
    output = tokenizer(dynamic)

    # 1 dynamic summary token + history_bins * max_patches_per_bin patches.
    assert output.tokens.shape == (2, 7, 48)
    assert output.mask.shape == (2, 7)
    assert output.mask[:, 0].all()
    assert (output.modality[:, 0] == tokenizer.SUMMARY).all()
    assert (output.modality[:, 1:] == tokenizer.DYNAMIC).all()
    # summary + one active patch in bin 0 + one active patch in bin 2.
    assert output.mask.sum(dim=1).tolist() == [3, 3]
    assert torch.isfinite(output.tokens).all()


def test_tokenizer_rejects_wrong_event_shape():
    tokenizer = SparseEventTokenizer(hidden_dim=32, output_dim=48, num_heads=4)
    bad = torch.zeros(1, 2, 32, 32)
    try:
        tokenizer(bad)
    except ValueError as exc:
        assert "[B, T, 2, H, W]" in str(exc)
    else:
        raise AssertionError("expected invalid event shape to fail")


def test_suffix_attention_blocks_keep_state_isolated_from_actions():
    # Layout: 3 prefix tokens, 1 state token, 20 action tokens with the
    # action chunk forming one bidirectional block.
    att_masks = torch.tensor(
        [[0, 0, 0, 1, 1] + [0] * 19], dtype=torch.float32
    )
    pad_masks = torch.ones(1, 24, dtype=torch.bool)
    att_2d = make_att_2d_masks(pad_masks, att_masks)

    # Prefix sees only the prefix.
    assert att_2d[0, :3, :3].all()
    assert not att_2d[0, :3, 3:].any()
    # State sees the prefix and itself, never the actions.
    assert att_2d[0, 3, :4].all()
    assert not att_2d[0, 3, 4:].any()
    # Actions see the prefix, the state and every other action.
    assert att_2d[0, 4:, :4].all()
    assert att_2d[0, 4:, 4:].all()
    # Bidirectional within the action block.
    assert att_2d[0, 4, 5] and att_2d[0, 5, 4]
