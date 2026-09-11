"""Sparse dynamic event tokenization for E-DynVLA.

This is deliberately not an object detector.  It converts a short history of
dynamic-object event voxels into a compact sequence of tokens.  The design
follows the useful part of event-language models: sparse spatio-temporal
selection, modality/type embeddings, temporal aggregation, and projection
into the VLM token space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EventTokenBatch:
    """Fixed-width token batch returned by :class:`SparseEventTokenizer`."""

    tokens: torch.Tensor
    mask: torch.Tensor
    density: torch.Tensor
    modality: torch.Tensor
    patch_index: torch.Tensor


class SparseEventTokenizer(nn.Module):
    """Tokenize dynamic event voxel histories without detection.

    Args:
        hidden_dim: Internal event-token width.
        output_dim: Width expected by the downstream VLM.
        patch_size: Non-overlapping spatial patch size.
        max_patches_per_bin: Highest-density patches kept for each time bin.
            Empty patches are masked after top-k selection.
        history_bins: Maximum supported temporal history.
        num_layers: Number of temporal Transformer encoder layers.
        num_heads: Attention heads in the temporal encoder.
        min_patch_density: Minimum mean absolute event activity for a valid
            sparse patch token.
    """

    DYNAMIC = 0
    SUMMARY = 1

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        output_dim: int = 960,
        patch_size: int = 16,
        max_patches_per_bin: int = 8,
        history_bins: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        min_patch_density: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if history_bins < 1 or max_patches_per_bin < 1:
            raise ValueError("history_bins and max_patches_per_bin must be positive")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.max_patches_per_bin = max_patches_per_bin
        self.history_bins = history_bins
        self.min_patch_density = min_patch_density

        self.patch_embed = nn.Conv2d(
            2, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.modality_embedding = nn.Embedding(2, hidden_dim)
        self.time_embedding = nn.Embedding(history_bins, hidden_dim)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.summary_tokens = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.summary_tokens, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    @property
    def max_tokens(self) -> int:
        return 1 + self.history_bins * self.max_patches_per_bin

    def forward(self, dynamic_voxels: torch.Tensor) -> EventTokenBatch:
        """Create sparse dynamic-event tokens.

        The input must have shape ``[B, T, 2, H, W]``.  Channel 0/1 stores
        OFF/ON event activity.  Values may be binary, counts, or
        confidence-weighted log counts.
        """
        self._validate_inputs(dynamic_voxels)
        batch_size = dynamic_voxels.shape[0]

        dynamic = self._tokenize_stream(dynamic_voxels)

        summary = self.summary_tokens[None].expand(batch_size, -1, -1)
        summary_modality = torch.full(
            (batch_size, 1),
            self.SUMMARY,
            device=summary.device,
            dtype=torch.long,
        )
        summary = summary + self.modality_embedding(summary_modality)
        summary_mask = torch.ones(
            batch_size, 1, device=summary.device, dtype=torch.bool
        )
        summary_density = torch.ones(
            batch_size, 1, device=summary.device, dtype=summary.dtype
        )
        summary_index = torch.full(
            (batch_size, 1), -1, device=summary.device, dtype=torch.long
        )

        tokens = torch.cat([summary, dynamic[0]], dim=1)
        mask = torch.cat([summary_mask, dynamic[1]], dim=1)
        density = torch.cat([summary_density, dynamic[2]], dim=1)
        modality = torch.cat([summary_modality, dynamic[3]], dim=1)
        patch_index = torch.cat([summary_index, dynamic[4]], dim=1)

        tokens = self.temporal_encoder(tokens, src_key_padding_mask=~mask)
        tokens = self.projector(self.output_norm(tokens))
        tokens = tokens.masked_fill(~mask[..., None], 0)
        return EventTokenBatch(tokens, mask, density, modality, patch_index)

    def _tokenize_stream(
        self, voxels: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        batch_size, time_bins, _, height, width = voxels.shape
        flat = voxels.reshape(batch_size * time_bins, 2, height, width)
        embedded = self.patch_embed(flat)
        _, hidden_dim, patch_h, patch_w = embedded.shape
        n_patches = patch_h * patch_w
        keep = min(self.max_patches_per_bin, n_patches)

        embedded = embedded.flatten(2).transpose(1, 2)
        density = voxels.abs().sum(dim=2).reshape(
            batch_size * time_bins, 1, height, width
        )
        density = torch.nn.functional.avg_pool2d(
            density, kernel_size=self.patch_size, stride=self.patch_size
        ).flatten(1)
        selected_density, selected_idx = density.topk(keep, dim=1, sorted=True)
        gather_idx = selected_idx[..., None].expand(-1, -1, hidden_dim)
        selected = embedded.gather(1, gather_idx)

        # Pad the patch dimension so configuration changes do not alter the VLM
        # prefix length when the input resolution contains fewer patches.
        if keep < self.max_patches_per_bin:
            pad_n = self.max_patches_per_bin - keep
            selected = torch.nn.functional.pad(selected, (0, 0, 0, pad_n))
            selected_density = torch.nn.functional.pad(
                selected_density, (0, pad_n)
            )
            selected_idx = torch.nn.functional.pad(
                selected_idx, (0, pad_n), value=-1
            )

        selected = selected.reshape(
            batch_size, time_bins, self.max_patches_per_bin, hidden_dim
        )
        selected_density = selected_density.reshape(
            batch_size, time_bins, self.max_patches_per_bin
        )
        selected_idx = selected_idx.reshape(
            batch_size, time_bins, self.max_patches_per_bin
        )
        mask = selected_density > self.min_patch_density

        time_idx = torch.arange(time_bins, device=voxels.device)
        time_emb = self.time_embedding(time_idx)[None, :, None, :]
        safe_idx = selected_idx.clamp_min(0)
        rows = torch.div(safe_idx, patch_w, rounding_mode="floor")
        cols = safe_idx.remainder(patch_w)
        coordinates = torch.stack(
            [
                (cols.to(voxels.dtype) + 0.5) / patch_w,
                (rows.to(voxels.dtype) + 0.5) / patch_h,
            ],
            dim=-1,
        )
        coord_emb = self.coordinate_embedding(coordinates)
        modality = torch.full_like(selected_idx, self.DYNAMIC)
        selected = (
            selected
            + time_emb
            + coord_emb
            + self.modality_embedding(modality)
        )

        sequence_len = time_bins * self.max_patches_per_bin
        return (
            selected.reshape(batch_size, sequence_len, hidden_dim),
            mask.reshape(batch_size, sequence_len),
            selected_density.reshape(batch_size, sequence_len),
            modality.reshape(batch_size, sequence_len),
            selected_idx.reshape(batch_size, sequence_len),
        )

    def _validate_inputs(self, dynamic_voxels: torch.Tensor) -> None:
        if dynamic_voxels.ndim != 5 or dynamic_voxels.shape[2] != 2:
            raise ValueError("event voxels must have shape [B, T, 2, H, W]")
        if dynamic_voxels.shape[1] > self.history_bins:
            raise ValueError(
                f"received {dynamic_voxels.shape[1]} bins, maximum is {self.history_bins}"
            )
        if dynamic_voxels.shape[-2] < self.patch_size or dynamic_voxels.shape[-1] < self.patch_size:
            raise ValueError("event voxel resolution is smaller than patch_size")
