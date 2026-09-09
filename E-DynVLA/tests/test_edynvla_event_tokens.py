import numpy as np
import torch

from policies.edynvla.data import voxelize_weighted_events
from policies.edynvla.event_tokenizer import SparseEventTokenizer
from policies.edynvla.event_wam import EventWAMHead, event_wam_loss


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


def test_sparse_tokenizer_keeps_static_dynamic_types_and_masks_empty_patches():
    tokenizer = SparseEventTokenizer(
        hidden_dim=32,
        output_dim=48,
        patch_size=8,
        max_patches_per_bin=2,
        history_bins=3,
        num_layers=1,
        num_heads=4,
    )
    static = torch.zeros(2, 3, 2, 16, 16)
    dynamic = torch.zeros_like(static)
    static[:, 0, 0, :8, :8] = 1
    dynamic[:, 2, 1, 8:, 8:] = 1
    output = tokenizer(static, dynamic)

    assert output.tokens.shape == (2, 14, 48)
    assert output.mask.shape == (2, 14)
    assert output.mask[:, :2].all()
    assert (output.modality[:, 2:8] == tokenizer.STATIC).all()
    assert (output.modality[:, 8:] == tokenizer.DYNAMIC).all()
    assert output.mask.sum(dim=1).tolist() == [4, 4]
    assert torch.isfinite(output.tokens).all()


def test_event_wam_shape_and_gradient():
    tokenizer = SparseEventTokenizer(
        hidden_dim=32,
        output_dim=48,
        patch_size=8,
        max_patches_per_bin=2,
        history_bins=2,
        num_layers=1,
        num_heads=4,
    )
    static = torch.rand(2, 2, 2, 16, 16)
    dynamic = torch.rand_like(static)
    token_batch = tokenizer(static, dynamic)
    wam = EventWAMHead(
        token_dim=48,
        hidden_dim=32,
        state_dim=7,
        action_dim=8,
        future_steps=3,
        output_channels=4,
        grid_size=(2, 3),
        num_layers=1,
        num_heads=4,
    )
    logits = wam(
        token_batch.tokens,
        token_batch.mask,
        torch.rand(2, 7),
        torch.rand(2, 4, 8),
    )
    assert logits.shape == (2, 3, 4, 2, 3)
    loss = event_wam_loss(logits, torch.zeros_like(logits))
    loss.backward()
    assert tokenizer.patch_embed.weight.grad is not None


def test_tokenizer_rejects_wrong_event_shape():
    tokenizer = SparseEventTokenizer(hidden_dim=32, output_dim=48, num_heads=4)
    bad = torch.zeros(1, 2, 32, 32)
    try:
        tokenizer(bad, bad)
    except ValueError as exc:
        assert "[B, T, 2, H, W]" in str(exc)
    else:
        raise AssertionError("expected invalid event shape to fail")
