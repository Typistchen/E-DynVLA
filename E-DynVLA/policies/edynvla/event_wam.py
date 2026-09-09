"""Lightweight event-world auxiliary head for E-DynVLA.

The head predicts future patch-level static/dynamic ON/OFF activity.  It is not
an RGB/video generator; its purpose is to force the policy representation to
model short-horizon world dynamics while keeping inference affordable.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EventWAMHead(nn.Module):
    """Predict future event activity from event, state, and action tokens."""

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
            memory = torch.cat([memory, self.action_proj(action_history)], dim=1)
            action_mask = torch.ones(
                action_history.shape[:2], dtype=torch.bool, device=state.device
            )
            memory_mask = torch.cat([memory_mask, action_mask], dim=1)

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
