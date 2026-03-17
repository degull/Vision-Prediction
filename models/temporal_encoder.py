import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Input:
        x: [B, T, D]
    Output:
        x + pe: [B, T, D]
    """

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, D], but got {x.shape}")

        T = x.size(1)
        if T > self.pe.size(1):
            raise ValueError(
                f"Sequence length T={T} exceeds max_len={self.pe.size(1)} "
                f"configured in SinusoidalPositionalEncoding."
            )

        return x + self.pe[:, :T, :]


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for frame-wise features.

    Input:
        frame_feats: [B, T, C_in]

    Output:
        seq_feats : [B, T, D]
        clip_feat : [B, D]

    Notes:
        - Uses TransformerEncoder as MVP temporal module
        - Later, this module can be replaced by Mamba / Hyena
    """

    def __init__(
        self,
        in_dim: int = 768,
        model_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 1536,
        dropout: float = 0.1,
        max_len: int = 16,
        pooling: str = "last",   # "last" or "mean"
    ):
        super().__init__()

        if pooling not in ["last", "mean"]:
            raise ValueError(f"pooling must be 'last' or 'mean', got {pooling}")

        self.in_dim = in_dim
        self.model_dim = model_dim
        self.pooling = pooling

        # Project input dim if needed
        self.input_proj = nn.Identity() if in_dim == model_dim else nn.Linear(in_dim, model_dim)

        self.pos_enc = SinusoidalPositionalEncoding(d_model=model_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, frame_feats: torch.Tensor):
        """
        Args:
            frame_feats: [B, T, C_in]

        Returns:
            seq_feats: [B, T, D]
            clip_feat: [B, D]
        """
        if frame_feats.dim() != 3:
            raise ValueError(
                f"Expected frame_feats to have shape [B, T, C], but got {frame_feats.shape}"
            )

        # [B, T, C_in] -> [B, T, D]
        x = self.input_proj(frame_feats)

        # Add temporal positional encoding
        x = self.pos_enc(x)

        # Temporal modeling
        x = self.encoder(x)

        # Final norm
        seq_feats = self.norm(x)  # [B, T, D]

        # Pool to clip-level summary
        if self.pooling == "last":
            clip_feat = seq_feats[:, -1, :]   # [B, D]
        else:
            clip_feat = seq_feats.mean(dim=1) # [B, D]

        return seq_feats, clip_feat