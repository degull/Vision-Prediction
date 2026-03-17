import os
import torch
import torch.nn as nn

from models.backbone import ConvNeXtV2Backbone
from models.temporal_encoder import TemporalEncoder
from models.multi_future_predictor import MultiFuturePredictor


class BranchEventHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, future_feats):
        if future_feats.dim() != 3:
            raise ValueError(f"Expected [B,K,D], got {future_feats.shape}")
        return self.head(future_feats)


class ClipBinaryHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_feat):
        if clip_feat.dim() != 2:
            raise ValueError(f"Expected [B,D], got {clip_feat.shape}")
        return self.head(clip_feat)


class ClipMultiClassHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, num_classes=4, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, clip_feat):
        if clip_feat.dim() != 2:
            raise ValueError(f"Expected [B,D], got {clip_feat.shape}")
        return self.head(clip_feat)


class MultiFutureCrossingDecisionPoseModel(nn.Module):
    """
    crossing (multi-future)
    + early decision (clip binary)
    + pose (clip multiclass)
    """
    def __init__(
        self,
        backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
        backbone_dim=768,
        temporal_dim=768,
        future_dim=256,
        num_futures=3,
        freeze_backbone=True,
        temporal_num_heads=8,
        temporal_num_layers=2,
        temporal_ff_dim=1536,
        temporal_dropout=0.1,
        temporal_max_len=16,
        temporal_pooling="last",
        future_hidden_dim=512,
        future_dropout=0.1,
        event_hidden_dim=128,
        event_dropout=0.1,
        early_hidden_dim=256,
        early_dropout=0.1,
        pose_hidden_dim=256,
        pose_dropout=0.1,
        pose_num_classes=4,
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

        self.early_head = ClipBinaryHead(
            in_dim=temporal_dim,
            hidden_dim=early_hidden_dim,
            dropout=early_dropout,
        )

        self.pose_head = ClipMultiClassHead(
            in_dim=temporal_dim,
            hidden_dim=pose_hidden_dim,
            num_classes=pose_num_classes,
            dropout=pose_dropout,
        )

        self.num_futures = num_futures
        self.pose_num_classes = pose_num_classes

    def load_temporal_checkpoint(
        self,
        ckpt_path,
        load_backbone=True,
        load_temporal=True,
        verbose=True,
    ):
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

    def forward(self, video):
        frame_feats = self.backbone(video)
        seq_feats, clip_feat = self.temporal_encoder(frame_feats)

        future_feats = self.future_predictor(clip_feat)
        branch_logits = self.branch_event_head(future_feats)
        branch_probs = torch.sigmoid(branch_logits)
        agg_prob, best_branch = branch_probs.max(dim=1)

        early_logit = self.early_head(clip_feat)
        early_prob = torch.sigmoid(early_logit)

        pose_logit = self.pose_head(clip_feat)  # [B,4]

        return {
            "frame_feats": frame_feats,
            "seq_feats": seq_feats,
            "clip_feat": clip_feat,
            "future_feats": future_feats,
            "branch_logits": branch_logits,
            "branch_probs": branch_probs,
            "agg_prob": agg_prob,
            "best_branch": best_branch,
            "early_logit": early_logit,
            "early_prob": early_prob,
            "pose_logit": pose_logit,
        }