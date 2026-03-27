# C:\Users\IIPL02\Desktop\Vision Prediction\models\temporal_event_model.py
# stage 1 model: backbone + frame encoder + temporal encoder + event head
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.backbone import ConvNeXtV2Backbone
except ModuleNotFoundError:
    from backbone import ConvNeXtV2Backbone

# ============================================================
# Utility
# ============================================================
def _print_shape(name: str, x: Optional[torch.Tensor], enabled: bool = False):
    if enabled and x is not None:
        print(f"[Shape] {name:30s}: {tuple(x.shape)}")


# ============================================================
# Volterra Layer
# ============================================================
class VolterraLayer(nn.Module):
    """
    Simple low-rank second-order interaction layer.

    Input : [*, D]
    Output: [*, D]

    y = x + alpha * Proj( (U x) ⊙ (V x) )
    """

    def __init__(self, dim: int, rank: int = 16, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.alpha = alpha

        self.u_proj = nn.Linear(dim, rank)
        self.v_proj = nn.Linear(dim, rank)
        self.out_proj = nn.Linear(rank, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.u_proj(x)                 # [*, R]
        v = self.v_proj(x)                 # [*, R]
        q = u * v                          # [*, R]
        q = self.dropout(q)
        q = self.out_proj(q)               # [*, D]
        return x + self.alpha * q


# ============================================================
# Transformer + Volterra Frame Encoder
# ============================================================
class TransformerVolterraBlock(nn.Module):
    """
    Token-level block for a single frame token sequence.

    Input : [Bf, N, D]
    Output: [Bf, N, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_dim: int = 1536,
        dropout: float = 0.1,
        use_volterra: bool = True,
        volterra_rank: int = 16,
        volterra_alpha: float = 1.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

        if use_volterra:
            self.volterra = VolterraLayer(
                dim=dim,
                rank=volterra_rank,
                alpha=volterra_alpha,
                dropout=dropout,
            )
        else:
            self.volterra = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Bf, N, D]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        x = self.volterra(x)
        return x


class TransformerVolterraFrameEncoder(nn.Module):
    """
    Operates on token sequence inside each frame.

    Input : [Bf, N, D]
    Output: [Bf, N, D]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 1536,
        dropout: float = 0.1,
        use_volterra: bool = True,
        volterra_rank: int = 16,
        volterra_alpha: float = 1.0,
        max_tokens: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.max_tokens = max_tokens

        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            TransformerVolterraBlock(
                dim=dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_volterra=use_volterra,
                volterra_rank=volterra_rank,
                volterra_alpha=volterra_alpha,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Bf, N, D]
        b, n, d = x.shape
        if n > self.max_tokens:
            raise ValueError(f"Token length {n} exceeds max_tokens={self.max_tokens}")

        x = x + self.pos_embed[:, :n, :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


# ============================================================
# Mamba-like temporal blocks
# ============================================================
class MambaStyleBlock(nn.Module):
    """
    Lightweight Mamba-style temporal mixer substitute.

    Input : [B, T, D]
    Output: [B, T, D]
    """

    def __init__(
        self,
        dim: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = dim * expand

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden_dim * 2)

        self.dw_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=hidden_dim,
        )

        self.state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.state_dim = state_dim
        self.conv_kernel = conv_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.norm(x)

        h, g = self.in_proj(x).chunk(2, dim=-1)   # [B, T, H], [B, T, H]

        h = h.transpose(1, 2)                     # [B, H, T]
        h = self.dw_conv(h)                       # [B, H, T + k - 1]
        h = h[:, :, :x.size(1)]                   # [B, H, T]
        h = h.transpose(1, 2)                     # [B, T, H]

        h = self.state_proj(h)
        h = F.silu(h)
        g = torch.sigmoid(g)

        y = h * g
        y = self.dropout(y)
        y = self.out_proj(y)

        return residual + y


class TemporalMambaScale(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaStyleBlock(
                dim=dim,
                state_dim=state_dim,
                conv_kernel=conv_kernel,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class TwoScaleTemporalMambaEncoder(nn.Module):
    """
    Scale 1: original frame-level sequence
    Scale 2: locally pooled/coarsened temporal sequence
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        fusion: str = "concat_proj",
        local_window: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.local_window = local_window
        self.fusion = fusion

        self.scale1 = TemporalMambaScale(
            dim=dim,
            num_layers=num_layers,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand=expand,
            dropout=dropout,
        )

        self.scale2 = TemporalMambaScale(
            dim=dim,
            num_layers=num_layers,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            expand=expand,
            dropout=dropout,
        )

        if fusion == "concat_proj":
            self.fuse = nn.Linear(dim * 2, dim)
        elif fusion == "add":
            self.fuse = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

        self.norm = nn.LayerNorm(dim)

    def local_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, d = x.shape
        w = self.local_window
        if w <= 1:
            return x

        pad_len = (w - (t % w)) % w
        if pad_len > 0:
            pad = x[:, -1:, :].repeat(1, pad_len, 1)
            x = torch.cat([x, pad], dim=1)

        t2 = x.size(1) // w
        x = x.view(b, t2, w, d).mean(dim=2)   # [B, T2, D]
        return x

    def upsample_to_length(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: [B, T2, D] -> [B, target_len, D]
        x = x.transpose(1, 2)  # [B, D, T2]
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        x = x.transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, D]
        _, t, _ = x.shape

        fine = self.scale1(x)                            # [B, T, D]
        coarse_in = self.local_pool(x)                   # [B, T2, D]
        coarse = self.scale2(coarse_in)                  # [B, T2, D]
        coarse_up = self.upsample_to_length(coarse, t)   # [B, T, D]

        if self.fusion == "concat_proj":
            fused = self.fuse(torch.cat([fine, coarse_up], dim=-1))
        else:
            fused = fine + coarse_up

        fused = self.norm(fused)

        return {
            "fine": fine,
            "coarse": coarse,
            "coarse_up": coarse_up,
            "fused": fused,
        }


# ============================================================
# Event Head
# ============================================================
class EventHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ============================================================
# Main Model
# ============================================================
class TemporalEventModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        backbone_dim: int = 768,
        freeze_backbone: bool = True,

        # Frame encoder
        frame_feature_dim: int = 768,
        frame_encoder_num_heads: int = 8,
        frame_encoder_num_layers: int = 2,
        frame_encoder_ff_dim: int = 1536,
        frame_encoder_dropout: float = 0.1,
        frame_encoder_use_volterra: bool = True,
        frame_encoder_volterra_rank: int = 16,
        frame_encoder_volterra_alpha: float = 1.0,

        # Temporal encoder
        temporal_encoder_type: str = "mamba_2scale",
        temporal_mamba_dim: int = 768,
        temporal_mamba_num_layers: int = 2,
        temporal_mamba_state_dim: int = 16,
        temporal_mamba_conv_kernel: int = 4,
        temporal_mamba_expand: int = 2,
        temporal_mamba_dropout: float = 0.1,
        temporal_mamba_fusion: str = "concat_proj",
        temporal_mamba_local_window: int = 4,
        temporal_pooling: str = "last",

        # Event head
        event_hidden_dim: int = 256,
        event_dropout: float = 0.1,

        # Debug
        debug_shapes: bool = False,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.backbone_dim = backbone_dim
        self.frame_feature_dim = frame_feature_dim
        self.temporal_feature_dim = temporal_mamba_dim
        self.temporal_pooling = temporal_pooling
        self.temporal_encoder_type = temporal_encoder_type
        self.debug_shapes = debug_shapes

        # ----------------------------------------------------
        # Spatial backbone (token output)
        # ----------------------------------------------------
        self.backbone = ConvNeXtV2Backbone(
            model_name=backbone_name,
            pretrained=True,
            freeze=freeze_backbone,
            output_mode="tokens",
        )

        # ----------------------------------------------------
        # Token dim projection: backbone_dim -> frame_feature_dim
        # Input token shape after backbone: [B, T, N, backbone_dim]
        # ----------------------------------------------------
        if backbone_dim != frame_feature_dim:
            self.frame_proj = nn.Linear(backbone_dim, frame_feature_dim)
        else:
            self.frame_proj = nn.Identity()

        # ----------------------------------------------------
        # Frame encoder: Transformer + Volterra on per-frame tokens
        # ----------------------------------------------------
        self.frame_encoder = TransformerVolterraFrameEncoder(
            dim=frame_feature_dim,
            num_heads=frame_encoder_num_heads,
            num_layers=frame_encoder_num_layers,
            ff_dim=frame_encoder_ff_dim,
            dropout=frame_encoder_dropout,
            use_volterra=frame_encoder_use_volterra,
            volterra_rank=frame_encoder_volterra_rank,
            volterra_alpha=frame_encoder_volterra_alpha,
            max_tokens=256,
        )

        # ----------------------------------------------------
        # Frame pooled repr -> temporal dim
        # ----------------------------------------------------
        if frame_feature_dim != temporal_mamba_dim:
            self.temporal_in_proj = nn.Linear(frame_feature_dim, temporal_mamba_dim)
        else:
            self.temporal_in_proj = nn.Identity()

        # ----------------------------------------------------
        # Temporal encoder
        # ----------------------------------------------------
        if temporal_encoder_type != "mamba_2scale":
            raise ValueError(
                f"Currently only temporal_encoder_type='mamba_2scale' is supported, "
                f"but got: {temporal_encoder_type}"
            )

        self.temporal_encoder = TwoScaleTemporalMambaEncoder(
            dim=temporal_mamba_dim,
            num_layers=temporal_mamba_num_layers,
            state_dim=temporal_mamba_state_dim,
            conv_kernel=temporal_mamba_conv_kernel,
            expand=temporal_mamba_expand,
            dropout=temporal_mamba_dropout,
            fusion=temporal_mamba_fusion,
            local_window=temporal_mamba_local_window,
        )

        # ----------------------------------------------------
        # Event head
        # ----------------------------------------------------
        self.event_head = EventHead(
            in_dim=temporal_mamba_dim,
            hidden_dim=event_hidden_dim,
            dropout=event_dropout,
        )

    def extract_frame_tokens(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B, T, 3, H, W]
        return: [B, T, N, backbone_dim]
        """
        tokens = self.backbone(video)
        return tokens

    def pool_frame_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*T, N, D]
        return: [B*T, D]
        """
        return x.mean(dim=1)

    def temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: [B, D]
        """
        if self.temporal_pooling == "last":
            return x[:, -1, :]
        elif self.temporal_pooling == "mean":
            return x.mean(dim=1)
        elif self.temporal_pooling == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"Unsupported temporal_pooling: {self.temporal_pooling}")

    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        video: [B, T, 3, H, W]
        """
        _print_shape("input_video", video, self.debug_shapes)

        # ----------------------------------------------------
        # [1] Spatial backbone -> tokens
        # [B, T, N, backbone_dim]
        # ----------------------------------------------------
        frame_tokens = self.extract_frame_tokens(video)
        _print_shape("frame_tokens_backbone", frame_tokens, self.debug_shapes)

        # ----------------------------------------------------
        # [2] Token projection
        # [B, T, N, frame_feature_dim]
        # ----------------------------------------------------
        frame_tokens = self.frame_proj(frame_tokens)
        _print_shape("frame_tokens_proj", frame_tokens, self.debug_shapes)

        b, t, n, d = frame_tokens.shape

        # ----------------------------------------------------
        # [3] Per-frame token encoder
        # [B, T, N, D] -> [B*T, N, D]
        # ----------------------------------------------------
        frame_tokens_bt = frame_tokens.view(b * t, n, d)
        _print_shape("frame_tokens_bt", frame_tokens_bt, self.debug_shapes)

        frame_encoded_bt = self.frame_encoder(frame_tokens_bt)
        _print_shape("frame_encoded_bt", frame_encoded_bt, self.debug_shapes)

        # ----------------------------------------------------
        # [4] Token -> frame representation
        # [B*T, N, D] -> [B*T, D] -> [B, T, D]
        # ----------------------------------------------------
        frame_pooled_bt = self.pool_frame_tokens(frame_encoded_bt)
        _print_shape("frame_pooled_bt", frame_pooled_bt, self.debug_shapes)

        frame_sequence = frame_pooled_bt.view(b, t, d)
        _print_shape("frame_sequence", frame_sequence, self.debug_shapes)

        # ----------------------------------------------------
        # [5] Frame repr -> temporal dim
        # [B, T, temporal_dim]
        # ----------------------------------------------------
        temporal_in = self.temporal_in_proj(frame_sequence)
        _print_shape("temporal_in", temporal_in, self.debug_shapes)

        # ----------------------------------------------------
        # [6] 2-scale temporal Mamba
        # ----------------------------------------------------
        temporal_dict = self.temporal_encoder(temporal_in)
        temporal_fused = temporal_dict["fused"]
        _print_shape("temporal_fused", temporal_fused, self.debug_shapes)

        # ----------------------------------------------------
        # [7] Temporal pooling
        # [B, temporal_dim]
        # ----------------------------------------------------
        pooled = self.temporal_pool(temporal_fused)
        _print_shape("pooled", pooled, self.debug_shapes)

        # ----------------------------------------------------
        # [8] Event head
        # [B, 1]
        # ----------------------------------------------------
        logits = self.event_head(pooled)
        _print_shape("logits", logits, self.debug_shapes)

        return {
            "logits": logits,

            # backbone / frame branch
            "frame_tokens": frame_tokens,                  # [B, T, N, D]
            "frame_encoded_tokens": frame_encoded_bt,      # [B*T, N, D]
            "frame_pooled_bt": frame_pooled_bt,            # [B*T, D]
            "frame_sequence": frame_sequence,              # [B, T, D]

            # temporal branch
            "temporal_in": temporal_in,
            "temporal_fine": temporal_dict["fine"],
            "temporal_coarse": temporal_dict["coarse"],
            "temporal_coarse_up": temporal_dict["coarse_up"],
            "temporal_fused": temporal_fused,

            # final pooled feature
            "pooled_feat": pooled,
        }