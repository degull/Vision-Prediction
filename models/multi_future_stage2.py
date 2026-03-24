# C:\Users\IIPL02\Desktop\Vision Prediction\models\multi_future_stage2.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal_event_model import TemporalEventModel


# ============================================================
# Utility
# ============================================================
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


# ============================================================
# Robust extraction from Stage-1 output
# ============================================================
def extract_stage1_pooled_feat(stage1_out):
    if not isinstance(stage1_out, dict):
        raise TypeError(
            f"TemporalEventModel output must be dict, but got {type(stage1_out)}"
        )

    candidate_keys = [
        "pooled_feat",
        "temporal_feat",
        "z",
        "feat",
        "features",
    ]

    for k in candidate_keys:
        if k in stage1_out:
            x = stage1_out[k]
            if not torch.is_tensor(x):
                continue
            if x.dim() == 2:
                return x
            if x.dim() == 3:
                return x.mean(dim=1)

    raise KeyError(
        "Could not find pooled feature in TemporalEventModel output. "
        f"Available keys: {list(stage1_out.keys())}"
    )


def extract_stage1_event_logit(stage1_out):
    if not isinstance(stage1_out, dict):
        raise TypeError(
            f"TemporalEventModel output must be dict, but got {type(stage1_out)}"
        )

    candidate_keys = [
        "event_logit",
        "logit",
        "logits",
        "crossing_logit",
        "event_logits",
        "crossing_logits",
        "output",
        "outputs",
    ]

    for k in candidate_keys:
        if k in stage1_out:
            x = stage1_out[k]
            if not torch.is_tensor(x):
                continue

            if x.dim() == 1:
                return x.unsqueeze(1)

            if x.dim() == 2:
                if x.size(1) == 1:
                    return x
                if x.size(1) == 2:
                    return (x[:, 1] - x[:, 0]).unsqueeze(1)

    raise KeyError(
        "Could not find event logit in TemporalEventModel output. "
        f"Available keys: {list(stage1_out.keys())}"
    )


# ============================================================
# Small building blocks
# ============================================================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ExpertBranch(nn.Module):
    """
    One context expert branch:
      input       : [B, D]
      output feat : [B, H]
      output logit: [B, 1]
      output risk : [B, 1]
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.event_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.trunk(x)
        event_logit = self.event_head(feat)
        risk_logit = self.risk_head(feat)
        return feat, event_logit, risk_logit


# ============================================================
# 2-Branch Context Expert Stage-2 Model
# ============================================================
class ContextExpertStage2Model(nn.Module):
    """
    Shared video encoder + structured context encoders + 2 expert branches

    Branch 1: Pedestrian-intent expert
        input = [z_video, z_attr, z_app]

    Branch 2: Environment-interaction expert
        input = [z_video, z_traffic, z_vehicle]

    Gating network predicts adaptive branch weights.

    final_logit =
        base_logit_weight * base_logit +
        expert_logit_weight * aggregated_expert_logit
    """

    def __init__(
        self,
        stage1_model: nn.Module = None,
        stage1_feat_dim: int = 768,

        attr_dim: int = 6,
        app_dim: int = 5,
        traffic_dim: int = 6,
        vehicle_dim: int = 6,

        context_embed_dim: int = 64,
        context_hidden_dim: int = 64,

        expert_hidden_dim: int = 128,
        gate_hidden_dim: int = 128,
        dropout: float = 0.1,

        base_logit_weight: float = 0.6,
        expert_logit_weight: float = 0.4,

        # ----------------------------------------------------
        # Stage1 config (must match train_temporal_event.py)
        # ----------------------------------------------------
        backbone_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        backbone_dim: int = 768,
        freeze_backbone: bool = True,

        frame_feature_dim: int = 768,
        frame_encoder_num_heads: int = 8,
        frame_encoder_num_layers: int = 2,
        frame_encoder_ff_dim: int = 1536,
        frame_encoder_dropout: float = 0.1,
        frame_encoder_use_volterra: bool = True,
        frame_encoder_volterra_rank: int = 16,
        frame_encoder_volterra_alpha: float = 1.0,

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

        event_hidden_dim: int = 256,
        event_dropout: float = 0.1,
    ):
        super().__init__()

        if stage1_model is None:
            stage1_model = TemporalEventModel(
                backbone_name=backbone_name,
                backbone_dim=backbone_dim,
                freeze_backbone=freeze_backbone,

                frame_feature_dim=frame_feature_dim,
                frame_encoder_num_heads=frame_encoder_num_heads,
                frame_encoder_num_layers=frame_encoder_num_layers,
                frame_encoder_ff_dim=frame_encoder_ff_dim,
                frame_encoder_dropout=frame_encoder_dropout,
                frame_encoder_use_volterra=frame_encoder_use_volterra,
                frame_encoder_volterra_rank=frame_encoder_volterra_rank,
                frame_encoder_volterra_alpha=frame_encoder_volterra_alpha,

                temporal_encoder_type=temporal_encoder_type,
                temporal_mamba_dim=temporal_mamba_dim,
                temporal_mamba_num_layers=temporal_mamba_num_layers,
                temporal_mamba_state_dim=temporal_mamba_state_dim,
                temporal_mamba_conv_kernel=temporal_mamba_conv_kernel,
                temporal_mamba_expand=temporal_mamba_expand,
                temporal_mamba_dropout=temporal_mamba_dropout,
                temporal_mamba_fusion=temporal_mamba_fusion,
                temporal_mamba_local_window=temporal_mamba_local_window,
                temporal_pooling=temporal_pooling,

                event_hidden_dim=event_hidden_dim,
                event_dropout=event_dropout,
            )

        self.stage1_model = stage1_model
        self.stage1_feat_dim = stage1_feat_dim

        self.attr_dim = attr_dim
        self.app_dim = app_dim
        self.traffic_dim = traffic_dim
        self.vehicle_dim = vehicle_dim

        if abs((base_logit_weight + expert_logit_weight) - 1.0) > 1e-6:
            raise ValueError("base_logit_weight + expert_logit_weight must sum to 1.0")

        self.base_logit_weight = float(base_logit_weight)
        self.expert_logit_weight = float(expert_logit_weight)

        # keep config for debugging / reproducibility
        self.stage1_config = {
            "backbone_name": backbone_name,
            "backbone_dim": backbone_dim,
            "freeze_backbone": freeze_backbone,
            "frame_feature_dim": frame_feature_dim,
            "frame_encoder_num_heads": frame_encoder_num_heads,
            "frame_encoder_num_layers": frame_encoder_num_layers,
            "frame_encoder_ff_dim": frame_encoder_ff_dim,
            "frame_encoder_dropout": frame_encoder_dropout,
            "frame_encoder_use_volterra": frame_encoder_use_volterra,
            "frame_encoder_volterra_rank": frame_encoder_volterra_rank,
            "frame_encoder_volterra_alpha": frame_encoder_volterra_alpha,
            "temporal_encoder_type": temporal_encoder_type,
            "temporal_mamba_dim": temporal_mamba_dim,
            "temporal_mamba_num_layers": temporal_mamba_num_layers,
            "temporal_mamba_state_dim": temporal_mamba_state_dim,
            "temporal_mamba_conv_kernel": temporal_mamba_conv_kernel,
            "temporal_mamba_expand": temporal_mamba_expand,
            "temporal_mamba_dropout": temporal_mamba_dropout,
            "temporal_mamba_fusion": temporal_mamba_fusion,
            "temporal_mamba_local_window": temporal_mamba_local_window,
            "temporal_pooling": temporal_pooling,
            "event_hidden_dim": event_hidden_dim,
            "event_dropout": event_dropout,
        }

        # ----------------------------------------------------
        # Context encoders
        # ----------------------------------------------------
        self.attr_encoder = MLPEncoder(attr_dim, context_hidden_dim, context_embed_dim, dropout=dropout)
        self.app_encoder = MLPEncoder(app_dim, context_hidden_dim, context_embed_dim, dropout=dropout)
        self.traffic_encoder = MLPEncoder(traffic_dim, context_hidden_dim, context_embed_dim, dropout=dropout)
        self.vehicle_encoder = MLPEncoder(vehicle_dim, context_hidden_dim, context_embed_dim, dropout=dropout)

        # ----------------------------------------------------
        # 2 expert branches
        # ----------------------------------------------------
        ped_input_dim = stage1_feat_dim + context_embed_dim + context_embed_dim
        env_input_dim = stage1_feat_dim + context_embed_dim + context_embed_dim

        self.pedestrian_expert = ExpertBranch(ped_input_dim, expert_hidden_dim, dropout=dropout)
        self.environment_expert = ExpertBranch(env_input_dim, expert_hidden_dim, dropout=dropout)

        # ----------------------------------------------------
        # Gating network
        # ----------------------------------------------------
        gate_input_dim = stage1_feat_dim + context_embed_dim * 4

        self.gating_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 2),
        )

    # --------------------------------------------------------
    # Freeze / unfreeze helpers
    # --------------------------------------------------------
    def freeze_stage1(self):
        set_requires_grad(self.stage1_model, False)

    def unfreeze_stage1(self):
        set_requires_grad(self.stage1_model, True)

    def freeze_stage1_except_keywords(self, keywords):
        keywords = [k.lower() for k in keywords]
        for name, p in self.stage1_model.named_parameters():
            low = name.lower()
            trainable = any(k in low for k in keywords)
            p.requires_grad = trainable

    def freeze_spatial_backbone_only(self):
        freeze_keywords = ["backbone", "convnext", "spatial"]
        for name, p in self.stage1_model.named_parameters():
            low = name.lower()
            if any(k in low for k in freeze_keywords):
                p.requires_grad = False

    def unfreeze_stage1_temporal_and_head_only(self):
        """
        ConvNeXt/backbone 쪽은 그대로 두고,
        temporal / transformer / volterra / mamba / classifier head 계열만 학습 허용
        """
        train_keywords = [
            "temporal",
            "transformer",
            "volterra",
            "mamba",
            "event_head",
            "head",
            "classifier",
            "fc",
            "norm",
            "fusion",
            "proj",
        ]
        freeze_keywords = ["backbone", "convnext", "spatial"]

        for name, p in self.stage1_model.named_parameters():
            low = name.lower()

            if any(k in low for k in freeze_keywords):
                p.requires_grad = False
                continue

            if any(k in low for k in train_keywords):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def print_trainable_summary(self):
        print("\n[2-Branch Context Expert Stage2 Trainable Parameters]")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name:90s} {tuple(p.shape)}")
        print(f"\n[Total Trainable Params] {count_trainable_params(self):,}\n")

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, video, attr_vec, app_vec, traffic_vec, vehicle_vec):
        stage1_out = self.stage1_model(video)

        z_video = extract_stage1_pooled_feat(stage1_out)      # [B, 768]
        base_logit = extract_stage1_event_logit(stage1_out)   # [B, 1]

        z_attr = self.attr_encoder(attr_vec)
        z_app = self.app_encoder(app_vec)
        z_traffic = self.traffic_encoder(traffic_vec)
        z_vehicle = self.vehicle_encoder(vehicle_vec)

        # Branch 1: Pedestrian-intent expert
        ped_input = torch.cat([z_video, z_attr, z_app], dim=1)
        ped_feat, ped_logit, ped_risk = self.pedestrian_expert(ped_input)

        # Branch 2: Environment-interaction expert
        env_input = torch.cat([z_video, z_traffic, z_vehicle], dim=1)
        env_feat, env_logit, env_risk = self.environment_expert(env_input)

        branch_features = torch.stack([ped_feat, env_feat], dim=1)    # [B,2,H]
        branch_logits = torch.stack([ped_logit, env_logit], dim=1)    # [B,2,1]
        branch_risks = torch.stack([ped_risk, env_risk], dim=1)       # [B,2,1]

        gate_input = torch.cat([z_video, z_attr, z_app, z_traffic, z_vehicle], dim=1)
        gate_logits = self.gating_network(gate_input)                 # [B,2]
        gate_weights = F.softmax(gate_logits, dim=1)                  # [B,2]

        aggregated_event_logit = torch.sum(
            gate_weights.unsqueeze(-1) * branch_logits, dim=1
        )  # [B,1]

        aggregated_risk_logit = torch.sum(
            gate_weights.unsqueeze(-1) * branch_risks, dim=1
        )  # [B,1]

        final_logit = (
            self.base_logit_weight * base_logit +
            self.expert_logit_weight * aggregated_event_logit
        )

        return {
            "stage1_out": stage1_out,

            "z_video": z_video,
            "z_attr": z_attr,
            "z_app": z_app,
            "z_traffic": z_traffic,
            "z_vehicle": z_vehicle,

            "base_logit": base_logit,

            "branch_features": branch_features,
            "branch_logits": branch_logits,
            "branch_risks": branch_risks,

            "gate_logits": gate_logits,
            "gate_weights": gate_weights,

            "aggregated_event_logit": aggregated_event_logit,
            "aggregated_risk_logit": aggregated_risk_logit,
            "final_logit": final_logit,
        }


# ============================================================
# Losses
# ============================================================
def binary_bce_with_logits(logits, labels, pos_weight=None, sample_weight=None):
    labels = labels.float()

    if pos_weight is not None:
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight), device=logits.device)
        pos_weight = pos_weight.to(logits.device).float()

    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )

    if sample_weight is not None:
        if not torch.is_tensor(sample_weight):
            sample_weight = torch.tensor(sample_weight, device=logits.device)
        sample_weight = sample_weight.to(logits.device).float()

        while sample_weight.dim() < loss.dim():
            sample_weight = sample_weight.unsqueeze(-1)

        loss = loss * sample_weight

    return loss.mean()


def final_classification_loss(final_logit, labels, pos_weight=None, sample_weight=None):
    return binary_bce_with_logits(
        final_logit, labels, pos_weight=pos_weight, sample_weight=sample_weight
    )


def branch_classification_loss(branch_logits, labels, pos_weight=None, sample_weight=None):
    labels_exp = labels.unsqueeze(1).expand(-1, branch_logits.size(1), -1)

    if sample_weight is not None:
        sample_weight = sample_weight.unsqueeze(1).expand(-1, branch_logits.size(1))
        sample_weight = sample_weight.reshape(-1)

    return binary_bce_with_logits(
        branch_logits.reshape(-1, 1),
        labels_exp.reshape(-1, 1),
        pos_weight=pos_weight,
        sample_weight=sample_weight,
    )


def distillation_loss(final_logit, base_logit):
    return F.mse_loss(
        torch.sigmoid(final_logit),
        torch.sigmoid(base_logit).detach(),
    )


def gate_balance_loss(gate_weights):
    usage = gate_weights.mean(dim=0)  # [2]
    target = torch.full_like(usage, 1.0 / usage.numel())
    return ((usage - target) ** 2).mean()


def branch_diversity_loss(branch_features):
    """
    branch feature cosine similarity를 낮추는 loss
    """
    B, K, D = branch_features.shape
    if K <= 1:
        return branch_features.new_tensor(0.0)

    x = F.normalize(branch_features, dim=-1)
    sim = torch.matmul(x, x.transpose(1, 2))  # [B,K,K]

    eye = torch.eye(K, device=sim.device).unsqueeze(0)
    offdiag = sim * (1.0 - eye)

    return (offdiag ** 2).sum() / max(B * (K * K - K), 1)


def branch_logit_margin_loss(branch_logits, gate_weights, labels, margin=0.20):
    """
    positive일 때는 적어도 한 expert가 강한 positive를 내고,
    negative일 때는 적어도 한 expert가 강한 negative를 내도록 유도
    """
    labels = labels.float().view(-1, 1)  # [B,1]
    logits = branch_logits.squeeze(-1)   # [B,K]

    pos_mask = (labels.squeeze(1) == 1)
    neg_mask = (labels.squeeze(1) == 0)

    loss = logits.new_tensor(0.0)

    if pos_mask.any():
        pos_logits = logits[pos_mask]  # [Bp,K]
        pos_best, _ = pos_logits.max(dim=1)
        loss = loss + F.relu(margin - pos_best).mean()

    if neg_mask.any():
        neg_logits = logits[neg_mask]  # [Bn,K]
        neg_best, _ = (-neg_logits).max(dim=1)
        loss = loss + F.relu(margin - neg_best).mean()

    return loss


def risk_regularization_loss(aggregated_risk_logit):
    return torch.mean(aggregated_risk_logit ** 2)


def compute_stage2_losses(
    outputs,
    labels,
    pos_weight=None,
    sample_weight=None,
    lambda_final=1.0,
    lambda_branch=0.3,
    lambda_distill=0.3,
    lambda_gate_balance=0.05,
    lambda_div=0.10,
    lambda_margin=0.10,
    lambda_risk_reg=0.01,
):
    final_logit = outputs["final_logit"]
    base_logit = outputs["base_logit"]
    branch_logits = outputs["branch_logits"]
    gate_weights = outputs["gate_weights"]
    branch_features = outputs["branch_features"]
    aggregated_risk_logit = outputs["aggregated_risk_logit"]

    loss_final = final_classification_loss(
        final_logit, labels, pos_weight=pos_weight, sample_weight=sample_weight
    )
    loss_branch = branch_classification_loss(
        branch_logits, labels, pos_weight=pos_weight, sample_weight=sample_weight
    )
    loss_distill = distillation_loss(final_logit, base_logit)
    loss_gate_balance = gate_balance_loss(gate_weights)
    loss_div = branch_diversity_loss(branch_features)
    loss_margin = branch_logit_margin_loss(branch_logits, gate_weights, labels)
    loss_risk_reg = risk_regularization_loss(aggregated_risk_logit)

    total = (
        lambda_final * loss_final +
        lambda_branch * loss_branch +
        lambda_distill * loss_distill +
        lambda_gate_balance * loss_gate_balance +
        lambda_div * loss_div +
        lambda_margin * loss_margin +
        lambda_risk_reg * loss_risk_reg
    )

    return {
        "total": total,
        "final": loss_final,
        "branch": loss_branch,
        "distill": loss_distill,
        "gate_balance": loss_gate_balance,
        "div": loss_div,
        "margin": loss_margin,
        "risk_reg": loss_risk_reg,
    }


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ContextExpertStage2Model(
        stage1_model=None,
        stage1_feat_dim=768,
        attr_dim=6,
        app_dim=5,
        traffic_dim=6,
        vehicle_dim=6,
        context_embed_dim=64,
        context_hidden_dim=64,
        expert_hidden_dim=128,
        gate_hidden_dim=128,
        dropout=0.1,
        base_logit_weight=0.6,
        expert_logit_weight=0.4,

        backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
        backbone_dim=768,
        freeze_backbone=True,

        frame_feature_dim=768,
        frame_encoder_num_heads=8,
        frame_encoder_num_layers=2,
        frame_encoder_ff_dim=1536,
        frame_encoder_dropout=0.1,
        frame_encoder_use_volterra=True,
        frame_encoder_volterra_rank=16,
        frame_encoder_volterra_alpha=1.0,

        temporal_encoder_type="mamba_2scale",
        temporal_mamba_dim=768,
        temporal_mamba_num_layers=2,
        temporal_mamba_state_dim=16,
        temporal_mamba_conv_kernel=4,
        temporal_mamba_expand=2,
        temporal_mamba_dropout=0.1,
        temporal_mamba_fusion="concat_proj",
        temporal_mamba_local_window=4,
        temporal_pooling="last",

        event_hidden_dim=256,
        event_dropout=0.1,
    ).to(device)

    model.freeze_stage1()

    x = torch.randn(2, 8, 3, 224, 224, device=device)
    attr_vec = torch.randn(2, 6, device=device)
    app_vec = torch.randn(2, 5, device=device)
    traffic_vec = torch.randn(2, 6, device=device)
    vehicle_vec = torch.randn(2, 6, device=device)
    y = torch.randint(0, 2, (2, 1), device=device).float()

    out = model(x, attr_vec, app_vec, traffic_vec, vehicle_vec)
    losses = compute_stage2_losses(out, y, pos_weight=1.0)

    print("[Smoke Test]")
    print("base_logit            :", tuple(out["base_logit"].shape))
    print("branch_logits         :", tuple(out["branch_logits"].shape))
    print("branch_risks          :", tuple(out["branch_risks"].shape))
    print("gate_weights          :", tuple(out["gate_weights"].shape))
    print("aggregated_event_logit:", tuple(out["aggregated_event_logit"].shape))
    print("final_logit           :", tuple(out["final_logit"].shape))
    print("loss_total            :", float(losses["total"].item()))
    print("trainable_params      :", count_trainable_params(model))
    print("stage1_config         :", model.stage1_config)