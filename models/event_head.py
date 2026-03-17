import torch
import torch.nn as nn


class EventHead(nn.Module):
    """
    Binary event head for crossing / not-crossing.

    Input:
        clip_feat: [B, D]

    Output:
        logits: [B, 1]
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_feat: torch.Tensor) -> torch.Tensor:
        if clip_feat.dim() != 2:
            raise ValueError(f"Expected clip_feat shape [B, D], got {clip_feat.shape}")
        return self.head(clip_feat)