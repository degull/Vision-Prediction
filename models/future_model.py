import os
import torch
import torch.nn as nn

from models.backbone import ConvNeXtV2Backbone
from models.temporal_encoder import TemporalEncoder
from models.future_model_decision import MultiFuturePredictor


class BranchEventHead(nn.Module):
    """
    Branch-wise binary event head.

    Input:
        future_feats: [B, K, D_f]

    Output:
        branch_logits: [B, K, 1]
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, future_feats: torch.Tensor) -> torch.Tensor:
        if future_feats.dim() != 3:
            raise ValueError(f"Expected future_feats shape [B, K, D], got {future_feats.shape}")
        return self.head(future_feats)  # [B, K, 1]


class MultiFutureCrossingModel(nn.Module):
    """
    Video -> Backbone -> Temporal Encoder -> MultiFuture Predictor -> Branch Event Head

    Input:
        video: [B, T, 3, H, W]

    Output dict:
        frame_feats   : [B, T, 768]
        seq_feats     : [B, T, 768]
        clip_feat     : [B, 768]
        future_feats  : [B, K, 256]
        branch_logits : [B, K, 1]
        branch_probs  : [B, K, 1]
        agg_prob      : [B, 1]   (max branch prob)
        best_branch   : [B, 1]   (argmax over K branches by prob)
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        backbone_dim: int = 768,
        temporal_dim: int = 768,
        future_dim: int = 256,
        num_futures: int = 3,
        freeze_backbone: bool = True,
        temporal_num_heads: int = 8,
        temporal_num_layers: int = 2,
        temporal_ff_dim: int = 1536,
        temporal_dropout: float = 0.1,
        temporal_max_len: int = 16,
        temporal_pooling: str = "last",
        future_hidden_dim: int = 512,
        future_dropout: float = 0.1,
        event_hidden_dim: int = 128,
        event_dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = ConvNeXtV2Backbone(
            model_name=backbone_name,
            pretrained=True,
            freeze=freeze_backbone,
        )

        self.temporal_encoder = TemporalEncoder(
            in_dim=backbone_dim,
            model_dim=temporal_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            ff_dim=temporal_ff_dim,
            dropout=temporal_dropout,
            max_len=temporal_max_len,
            pooling=temporal_pooling,
        )

        self.future_predictor = MultiFuturePredictor(
            in_dim=temporal_dim,
            future_dim=future_dim,
            num_futures=num_futures,
            hidden_dim=future_hidden_dim,
            dropout=future_dropout,
        )

        self.branch_event_head = BranchEventHead(
            in_dim=future_dim,
            hidden_dim=event_hidden_dim,
            dropout=event_dropout,
        )

        self.num_futures = num_futures

    def load_temporal_checkpoint(
        self,
        ckpt_path: str,
        load_backbone: bool = True,
        load_temporal: bool = True,
        verbose: bool = True,
    ):
        """
        Load only backbone / temporal_encoder weights from stage-1 temporal checkpoint.
        Does NOT load old single event head.

        Returns:
            dict with load summary
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" not in ckpt:
            raise KeyError("Checkpoint missing model_state_dict")

        src_state = ckpt["model_state_dict"]
        dst_state = self.state_dict()

        load_state = {}
        loaded_keys = []

        for k, v in src_state.items():
            if load_backbone and k.startswith("backbone.") and k in dst_state and dst_state[k].shape == v.shape:
                load_state[k] = v
                loaded_keys.append(k)

            if load_temporal and k.startswith("temporal_encoder.") and k in dst_state and dst_state[k].shape == v.shape:
                load_state[k] = v
                loaded_keys.append(k)

        missing, unexpected = self.load_state_dict(load_state, strict=False)

        if verbose:
            num_backbone = sum(1 for k in loaded_keys if k.startswith("backbone."))
            num_temporal = sum(1 for k in loaded_keys if k.startswith("temporal_encoder."))

            print("\n[Load Temporal Checkpoint]")
            print(f"  ckpt_path         : {ckpt_path}")
            print(f"  loaded backbone   : {num_backbone} keys")
            print(f"  loaded temporal   : {num_temporal} keys")
            print(f"  missing keys      : {len(missing)}")
            print(f"  unexpected keys   : {len(unexpected)}")

        return {
            "loaded_keys": loaded_keys,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }

    def forward(self, video: torch.Tensor):
        frame_feats = self.backbone(video)                         # [B, T, 768]
        seq_feats, clip_feat = self.temporal_encoder(frame_feats) # [B, T, 768], [B, 768]
        future_feats = self.future_predictor(clip_feat)           # [B, K, 256]
        branch_logits = self.branch_event_head(future_feats)      # [B, K, 1]
        branch_probs = torch.sigmoid(branch_logits)               # [B, K, 1]

        agg_prob, best_branch = branch_probs.max(dim=1)           # [B, 1], [B, 1]

        return {
            "frame_feats": frame_feats,
            "seq_feats": seq_feats,
            "clip_feat": clip_feat,
            "future_feats": future_feats,
            "branch_logits": branch_logits,
            "branch_probs": branch_probs,
            "agg_prob": agg_prob,
            "best_branch": best_branch,
        }