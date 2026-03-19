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
class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class FutureBranch(nn.Module):
    """
    Lighter branch:
      input  : [B, F]
      output :
        feat       [B, F]
        event_logit[B, 1]
        conf_logit [B, 1]
    """

    def __init__(self, future_dim: int, mlp_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.residual_mlp = MLPBlock(future_dim, mlp_hidden_dim, dropout=dropout)
        self.event_head = nn.Linear(future_dim, 1)
        self.conf_head = nn.Linear(future_dim, 1)

    def forward(self, x):
        feat = x + self.residual_mlp(x)
        event_logit = self.event_head(feat)
        conf_logit = self.conf_head(feat)
        return feat, event_logit, conf_logit


# ============================================================
# Main Stage-2 model
# ============================================================
class MultiFutureStage2Model(nn.Module):
    """
    Stronger base-preserving 2-branch default version.

    final_logit = base_w * base_logit + future_w * future_logit
    with stronger base preservation by default.
    """

    def __init__(
        self,
        stage1_model: nn.Module = None,
        stage1_feat_dim: int = 768,
        adapter_hidden_dim: int = 256,
        future_dim: int = 128,
        shared_hidden_dim: int = 128,
        branch_hidden_dim: int = 128,
        num_branches: int = 2,
        dropout: float = 0.1,
        base_logit_weight: float = 0.85,
        future_logit_weight: float = 0.15,
    ):
        super().__init__()

        if stage1_model is None:
            stage1_model = TemporalEventModel()

        self.stage1_model = stage1_model
        self.stage1_feat_dim = stage1_feat_dim
        self.future_dim = future_dim
        self.num_branches = num_branches

        if abs((base_logit_weight + future_logit_weight) - 1.0) > 1e-6:
            raise ValueError("base_logit_weight + future_logit_weight must sum to 1.0")

        self.base_logit_weight = float(base_logit_weight)
        self.future_logit_weight = float(future_logit_weight)

        # Adapter
        self.adapter = nn.Sequential(
            nn.Linear(stage1_feat_dim, adapter_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_dim, future_dim),
            nn.ReLU(inplace=True),
        )

        # Shared future trunk
        self.shared_future = nn.Sequential(
            nn.Linear(future_dim, shared_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, future_dim),
            nn.ReLU(inplace=True),
        )

        self.branch_embeddings = nn.Parameter(
            torch.randn(num_branches, future_dim) * 0.02
        )

        self.branches = nn.ModuleList([
            FutureBranch(
                future_dim=future_dim,
                mlp_hidden_dim=branch_hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_branches)
        ])

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

    def print_trainable_summary(self):
        print("\n[Stage2 Trainable Parameters]")
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name:90s} {tuple(p.shape)}")
        print(f"\n[Total Trainable Params] {count_trainable_params(self):,}\n")

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, video):
        stage1_out = self.stage1_model(video)

        z_base = extract_stage1_pooled_feat(stage1_out)      # [B, D]
        base_logit = extract_stage1_event_logit(stage1_out)  # [B, 1]

        z_adapter = self.adapter(z_base)         # [B, F]
        z_shared = self.shared_future(z_adapter) # [B, F]

        branch_features = []
        branch_logits = []
        branch_conf_logits = []

        for k, branch in enumerate(self.branches):
            xk = z_shared + self.branch_embeddings[k].unsqueeze(0)
            feat_k, logit_k, conf_k = branch(xk)

            branch_features.append(feat_k)
            branch_logits.append(logit_k)
            branch_conf_logits.append(conf_k)

        branch_features = torch.stack(branch_features, dim=1)         # [B, K, F]
        branch_logits = torch.stack(branch_logits, dim=1)             # [B, K, 1]
        branch_conf_logits = torch.stack(branch_conf_logits, dim=1)   # [B, K, 1]

        branch_conf = F.softmax(branch_conf_logits.squeeze(-1), dim=1)   # [B, K]
        future_logit = torch.sum(branch_conf.unsqueeze(-1) * branch_logits, dim=1)  # [B,1]

        final_logit = (
            self.base_logit_weight * base_logit +
            self.future_logit_weight * future_logit
        )

        return {
            "stage1_out": stage1_out,
            "z_base": z_base,
            "z_adapter": z_adapter,
            "z_shared": z_shared,
            "base_logit": base_logit,
            "branch_features": branch_features,
            "branch_logits": branch_logits,
            "branch_conf_logits": branch_conf_logits,
            "branch_conf": branch_conf,
            "future_logit": future_logit,
            "final_logit": final_logit,
        }


# ============================================================
# Losses
# ============================================================
def balanced_bce_loss(logits, labels, pos_weight_min=0.5, pos_weight_max=2.0):
    labels = labels.float()

    pos = (labels == 1).float().sum()
    neg = (labels == 0).float().sum()

    pos_weight = neg / pos.clamp_min(1.0)
    pos_weight = pos_weight.clamp(min=pos_weight_min, max=pos_weight_max).to(logits.device)

    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
    )


def final_classification_loss(final_logit, labels):
    return balanced_bce_loss(final_logit, labels)


def branch_classification_loss(branch_logits, labels):
    labels_exp = labels.unsqueeze(1).expand(-1, branch_logits.size(1), -1)
    return balanced_bce_loss(
        branch_logits.reshape(-1, 1),
        labels_exp.reshape(-1, 1),
    )


def branch_balance_loss(branch_conf):
    usage = branch_conf.mean(dim=0)    # [K]
    target = torch.full_like(usage, 1.0 / usage.numel())
    return ((usage - target) ** 2).mean()


def branch_diversity_loss(branch_features):
    B, K, Fdim = branch_features.shape
    if K <= 1:
        return branch_features.new_tensor(0.0)

    x = F.normalize(branch_features, dim=-1)
    sim = torch.matmul(x, x.transpose(1, 2))           # [B,K,K]

    eye = torch.eye(K, device=sim.device).unsqueeze(0)
    offdiag = sim * (1.0 - eye)

    loss = F.relu(offdiag).sum() / max(B * (K * K - K), 1)
    return loss


def distillation_loss(final_logit, base_logit):
    return F.mse_loss(
        torch.sigmoid(final_logit),
        torch.sigmoid(base_logit).detach(),
    )


def compute_stage2_losses(
    outputs,
    labels,
    lambda_final=1.0,
    lambda_branch=0.05,
    lambda_balance=0.01,
    lambda_div=0.005,
    lambda_distill=1.0,
):
    final_logit = outputs["final_logit"]
    base_logit = outputs["base_logit"]
    branch_logits = outputs["branch_logits"]
    branch_features = outputs["branch_features"]
    branch_conf = outputs["branch_conf"]

    loss_final = final_classification_loss(final_logit, labels)
    loss_branch = branch_classification_loss(branch_logits, labels)
    loss_balance = branch_balance_loss(branch_conf)
    loss_div = branch_diversity_loss(branch_features)
    loss_distill = distillation_loss(final_logit, base_logit)

    total = (
        lambda_final * loss_final +
        lambda_branch * loss_branch +
        lambda_balance * loss_balance +
        lambda_div * loss_div +
        lambda_distill * loss_distill
    )

    return {
        "total": total,
        "final": loss_final,
        "branch": loss_branch,
        "balance": loss_balance,
        "div": loss_div,
        "distill": loss_distill,
    }


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiFutureStage2Model(
        stage1_model=TemporalEventModel(),
        stage1_feat_dim=768,
        adapter_hidden_dim=256,
        future_dim=128,
        shared_hidden_dim=128,
        branch_hidden_dim=128,
        num_branches=2,
        dropout=0.1,
        base_logit_weight=0.85,
        future_logit_weight=0.15,
    ).to(device)

    model.freeze_stage1()

    x = torch.randn(2, 8, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (2, 1), device=device).float()

    out = model(x)
    losses = compute_stage2_losses(out, y)

    print("[Smoke Test]")
    print("base_logit     :", tuple(out["base_logit"].shape))
    print("branch_logits  :", tuple(out["branch_logits"].shape))
    print("branch_conf    :", tuple(out["branch_conf"].shape))
    print("future_logit   :", tuple(out["future_logit"].shape))
    print("final_logit    :", tuple(out["final_logit"].shape))
    print("loss_total     :", float(losses["total"].item()))
    print("trainable_params:", count_trainable_params(model))