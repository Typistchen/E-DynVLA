"""Lightweight multimodal world-action heads for E-DynVLA.

The main head predicts future RGB patches and static/dynamic ON/OFF event
activity. It operates on a compact grid rather than generating full-resolution
video, keeping the auxiliary world objective affordable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class WAMOutput:
    """Action-conditioned predictions of the future visual world."""

    rgb: torch.Tensor
    event_logits: torch.Tensor


class WorldActionModelHead(nn.Module):
    """Predict future RGB patches and event activity from multimodal context.

    RGB is predicted once at the configured horizon on a low-resolution patch
    grid. Events retain the finer temporal resolution of ``future_steps``.
    """

    def __init__(
        self,
        *,
        context_dim: int = 768,
        hidden_dim: int = 256,
        action_dim: int = 32,
        future_steps: int = 10,
        event_channels: int = 4,
        rgb_channels: int = 3,
        grid_size: tuple[int, int] = (12, 16),
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.future_steps = future_steps
        self.event_channels = event_channels
        self.rgb_channels = rgb_channels
        self.grid_size = tuple(grid_size)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        # Query 0 predicts RGB at the horizon; the rest predict event bins.
        self.future_queries = nn.Parameter(torch.empty(future_steps + 1, hidden_dim))
        nn.init.normal_(self.future_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        grid_cells = self.grid_size[0] * self.grid_size[1]
        self.rgb_output = nn.Linear(hidden_dim, rgb_channels * grid_cells)
        self.event_output = nn.Linear(hidden_dim, event_channels * grid_cells)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        action_history: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> WAMOutput:
        if context_tokens.ndim != 3 or context_mask.shape != context_tokens.shape[:2]:
            raise ValueError("context_tokens/mask must have shapes [B,L,D] and [B,L]")
        if action_history.ndim != 3 or action_history.shape[0] != context_tokens.shape[0]:
            raise ValueError("action_history must have shape [B,A,action_dim]")
        if action_mask is None:
            action_mask = torch.ones(
                action_history.shape[:2], dtype=torch.bool, device=action_history.device
            )
        elif action_mask.shape != action_history.shape[:2]:
            raise ValueError("action_mask must have shape [B,A]")
        else:
            action_mask = action_mask.to(device=action_history.device, dtype=torch.bool)

        memory = torch.cat(
            [self.context_proj(context_tokens), self.action_proj(action_history)], dim=1
        )
        memory_mask = torch.cat([context_mask.bool(), action_mask], dim=1)
        queries = self.future_queries[None].expand(context_tokens.shape[0], -1, -1)
        decoded = self.decoder(
            queries, memory, memory_key_padding_mask=~memory_mask
        )
        grid_h, grid_w = self.grid_size
        rgb = self.rgb_output(decoded[:, 0]).reshape(
            context_tokens.shape[0], self.rgb_channels, grid_h, grid_w
        ).sigmoid()
        event_logits = self.event_output(decoded[:, 1:]).reshape(
            context_tokens.shape[0],
            self.future_steps,
            self.event_channels,
            grid_h,
            grid_w,
        )
        return WAMOutput(rgb=rgb, event_logits=event_logits)


def multimodal_wam_loss(
    output: WAMOutput,
    target_rgb: torch.Tensor,
    target_events: torch.Tensor,
    *,
    rgb_weight: float = 1.0,
    event_weight: float = 1.0,
    positive_weight: float = 4.0,
    rgb_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint low-resolution RGB reconstruction and sparse event loss."""
    if output.rgb.shape != target_rgb.shape:
        raise ValueError("WAM RGB prediction and target must have equal shapes")
    rgb_error = F.smooth_l1_loss(
        output.rgb, target_rgb.to(output.rgb.dtype), reduction="none"
    ).mean(dim=(1, 2, 3))
    if rgb_valid_mask is None:
        rgb_loss = rgb_error.mean()
    else:
        valid = rgb_valid_mask.to(device=rgb_error.device, dtype=rgb_error.dtype).flatten()
        if valid.shape != rgb_error.shape:
            raise ValueError("rgb_valid_mask must have shape [B]")
        rgb_loss = (rgb_error * valid).sum() / valid.sum().clamp_min(1.0)
    event_loss = event_wam_loss(
        output.event_logits, target_events, positive_weight=positive_weight
    )
    total = rgb_weight * rgb_loss + event_weight * event_loss
    return total, {"rgb_loss": rgb_loss, "event_loss": event_loss}


class EventWAMHead(nn.Module):
    """Legacy event-only prediction head kept for checkpoint compatibility."""

    def __init__(
        self,
        *,
        token_dim: int = 768,
        hidden_dim: int = 256,
        state_dim: int = 32,
        action_dim: int = 32,
        future_steps: int = 10,
        output_channels: int = 4,
        grid_size: tuple[int, int] = (12, 16),
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.future_steps = future_steps
        self.output_channels = output_channels
        self.grid_size = tuple(grid_size)
        self.memory_proj = nn.Linear(token_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.future_queries = nn.Parameter(torch.empty(future_steps, hidden_dim))
        nn.init.normal_(self.future_queries, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(
            hidden_dim, output_channels * self.grid_size[0] * self.grid_size[1]
        )

    def forward(
        self,
        event_tokens: torch.Tensor,
        event_mask: torch.Tensor,
        state: torch.Tensor,
        action_history: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits shaped ``[B, F, 4, grid_h, grid_w]``.

        The four channels are static-OFF, static-ON, dynamic-OFF, dynamic-ON.
        """
        if event_tokens.ndim != 3 or event_mask.shape != event_tokens.shape[:2]:
            raise ValueError("event_tokens/mask must have shapes [B,L,D] and [B,L]")
        if state.ndim != 2:
            raise ValueError("state must have shape [B,state_dim]")

        memory = self.memory_proj(event_tokens)
        state_token = self.state_proj(state)[:, None]
        memory = torch.cat([memory, state_token], dim=1)
        memory_mask = torch.cat(
            [
                event_mask.bool(),
                torch.ones(state.shape[0], 1, dtype=torch.bool, device=state.device),
            ],
            dim=1,
        )
        if action_history is not None:
            if action_history.ndim != 3:
                raise ValueError("action_history must have shape [B,A,action_dim]")
            if action_history.shape[0] != state.shape[0]:
                raise ValueError("action_history and state batch sizes must match")
            memory = torch.cat([memory, self.action_proj(action_history)], dim=1)
            if action_mask is None:
                action_mask = torch.ones(
                    action_history.shape[:2], dtype=torch.bool, device=state.device
                )
            elif action_mask.shape != action_history.shape[:2]:
                raise ValueError("action_mask must have shape [B,A]")
            else:
                action_mask = action_mask.to(device=state.device, dtype=torch.bool)
            memory_mask = torch.cat([memory_mask, action_mask], dim=1)
        elif action_mask is not None:
            raise ValueError("action_mask requires action_history")

        queries = self.future_queries[None].expand(state.shape[0], -1, -1)
        decoded = self.decoder(
            queries, memory, memory_key_padding_mask=~memory_mask
        )
        logits = self.output(decoded)
        return logits.reshape(
            state.shape[0],
            self.future_steps,
            self.output_channels,
            self.grid_size[0],
            self.grid_size[1],
        )


def event_wam_loss(
    logits: torch.Tensor,
    target_activity: torch.Tensor,
    *,
    positive_weight: float = 4.0,
) -> torch.Tensor:
    """Sparse binary activity loss for the future-event auxiliary objective."""
    if logits.shape != target_activity.shape:
        raise ValueError("WAM logits and target_activity must have equal shapes")
    target_activity = target_activity.to(dtype=logits.dtype)
    pos_weight = torch.tensor(
        positive_weight, device=logits.device, dtype=logits.dtype
    )
    return F.binary_cross_entropy_with_logits(
        logits, target_activity, pos_weight=pos_weight
    )


@torch.no_grad()
def event_wam_metrics(
    logits: torch.Tensor,
    target_activity: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return sparse-event precision, recall, F1, IoU, and probability MAE."""
    if logits.shape != target_activity.shape:
        raise ValueError("WAM logits and target_activity must have equal shapes")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    probabilities = logits.sigmoid()
    prediction = probabilities >= threshold
    target = target_activity.bool()
    tp = (prediction & target).sum(dtype=torch.float32)
    fp = (prediction & ~target).sum(dtype=torch.float32)
    fn = (~prediction & target).sum(dtype=torch.float32)
    eps = torch.finfo(torch.float32).eps
    precision = tp / (tp + fp).clamp_min(eps)
    recall = tp / (tp + fn).clamp_min(eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall).clamp_min(eps),
        "iou": tp / (tp + fp + fn).clamp_min(eps),
        "probability_mae": (probabilities - target_activity.float()).abs().mean(),
    }


@torch.no_grad()
def multimodal_wam_metrics(
    output: WAMOutput,
    target_rgb: torch.Tensor,
    target_events: torch.Tensor,
    *,
    rgb_valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Metrics for both WAM outputs, suitable for TensorBoard/W&B logging."""
    rgb_error = output.rgb - target_rgb.to(output.rgb.dtype)
    if rgb_valid_mask is not None:
        valid = rgb_valid_mask.to(device=rgb_error.device, dtype=torch.bool).flatten()
        if valid.shape != rgb_error.shape[:1]:
            raise ValueError("rgb_valid_mask must have shape [B]")
        rgb_error = rgb_error[valid]
    if rgb_error.numel():
        rgb_mae = rgb_error.abs().mean()
        rgb_mse = rgb_error.square().mean()
    else:
        rgb_mae = output.rgb.new_zeros(())
        rgb_mse = output.rgb.new_zeros(())
    metrics = {
        "rgb_mae": rgb_mae,
        "rgb_psnr_db": -10.0 * torch.log10(rgb_mse.clamp_min(1e-8)),
    }
    metrics.update(
        {
            f"event_{name}": value
            for name, value in event_wam_metrics(
                output.event_logits, target_events
            ).items()
        }
    )
    return metrics
