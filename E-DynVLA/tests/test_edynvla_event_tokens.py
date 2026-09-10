import numpy as np
import torch

from policies.edynvla.data import (
    voxelize_weighted_event_pair,
    voxelize_weighted_events,
)
from policies.edynvla.event_tokenizer import SparseEventTokenizer
from policies.edynvla.event_wam import (
    EventWAMHead,
    WorldActionModelHead,
    event_wam_loss,
    event_wam_metrics,
    multimodal_wam_loss,
    multimodal_wam_metrics,
)


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


def test_multimodal_wam_predicts_rgb_and_event_and_backpropagates():
    wam = WorldActionModelHead(
        context_dim=48,
        hidden_dim=32,
        action_dim=8,
        future_steps=3,
        grid_size=(2, 3),
        num_layers=1,
        num_heads=4,
    )
    context = torch.rand(2, 7, 48, requires_grad=True)
    output = wam(
        context,
        torch.ones(2, 7, dtype=torch.bool),
        torch.rand(2, 4, 8),
        action_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
    )
    assert output.rgb.shape == (2, 3, 2, 3)
    assert output.event_logits.shape == (2, 3, 4, 2, 3)
    assert output.rgb.min() >= 0 and output.rgb.max() <= 1

    loss, parts = multimodal_wam_loss(
        output,
        torch.rand_like(output.rgb),
        torch.zeros_like(output.event_logits),
        rgb_valid_mask=torch.tensor([True, False]),
    )
    loss.backward()
    assert set(parts) == {"rgb_loss", "event_loss"}
    assert context.grad is not None
    metrics = multimodal_wam_metrics(
        output,
        torch.rand_like(output.rgb),
        torch.zeros_like(output.event_logits),
        rgb_valid_mask=torch.tensor([True, False]),
    )
    assert "rgb_psnr_db" in metrics
    assert "event_f1" in metrics


def test_event_wam_metrics_are_exact_for_separable_logits():
    target = torch.tensor([[[[[0.0, 1.0]]]]])
    logits = torch.tensor([[[[[-10.0, 10.0]]]]])
    metrics = event_wam_metrics(logits, target)
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["f1"] == 1
    assert metrics["iou"] == 1
