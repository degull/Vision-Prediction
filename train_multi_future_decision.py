import os
import csv
import time
import random
import argparse
import inspect
import importlib
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models.temporal_event_model import TemporalEventModel


# ============================================================
# Utility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_modules(model):
    print("\n[Trainable Parameters]")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:80s} {tuple(p.shape)}")
    print()


# ============================================================
# Dynamic Dataset Resolver
# ============================================================
def resolve_dataset_class():
    """
    Try multiple dataset modules / class names so that
    import errors do not occur even if the project structure differs.
    """
    candidates = [
        ("datasets.jaad_video_dataset", ["JAADVideoDataset", "JAADCrossingClipDataset"]),
        ("datasets.jaad_crossing_clip_dataset", ["JAADCrossingClipDataset", "JAADVideoDataset"]),
        ("jaad_video_dataset", ["JAADVideoDataset", "JAADCrossingClipDataset"]),
        ("jaad_crossing_clip_dataset", ["JAADCrossingClipDataset", "JAADVideoDataset"]),
    ]

    errors = []

    for module_name, class_names in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            errors.append(f"{module_name}: {repr(e)}")
            continue

        for cls_name in class_names:
            if hasattr(module, cls_name):
                cls = getattr(module, cls_name)
                print(f"[Dataset Resolved] module={module_name} | class={cls_name}")
                return cls

    msg = "\n".join(errors)
    raise ImportError(
        "Could not resolve JAAD dataset class.\n"
        "Tried modules/classes:\n"
        f"{msg}"
    )


def build_dataset(dataset_cls, args):
    """
    Pass only supported kwargs based on actual __init__ signature.
    """
    sig = inspect.signature(dataset_cls.__init__)
    supported = set(sig.parameters.keys())

    candidate_kwargs = {
        "clips_dir": args.clips_dir,
        "annotations_dir": args.annotations_dir,
        "attributes_dir": args.attributes_dir,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "frame_stride": args.frame_stride,
        "sample_stride": args.sample_stride,
        "early_horizon": args.early_horizon,
        "verbose": True,
    }

    final_kwargs = {}
    for k, v in candidate_kwargs.items():
        if k in supported:
            final_kwargs[k] = v

    print("[Dataset Init Args]")
    for k, v in final_kwargs.items():
        print(f"  {k}: {v}")
    print()

    dataset = dataset_cls(**final_kwargs)
    return dataset


def infer_video_and_label_keys(sample):
    """
    Support different dataset return dict formats.
    """
    if not isinstance(sample, dict):
        raise TypeError("Dataset sample must be a dict for this training script.")

    video_key_candidates = ["video", "frames", "clip", "images"]
    label_key_candidates = ["crossing_label", "crossing", "label", "target", "y"]

    video_key = None
    label_key = None

    for k in video_key_candidates:
        if k in sample:
            video_key = k
            break

    for k in label_key_candidates:
        if k in sample:
            label_key = k
            break

    if video_key is None:
        raise KeyError(f"Could not infer video key from sample keys: {list(sample.keys())}")
    if label_key is None:
        raise KeyError(f"Could not infer label key from sample keys: {list(sample.keys())}")

    print(f"[Detected Sample Keys] video_key={video_key} | label_key={label_key}")
    return video_key, label_key


# ============================================================
# Metrics
# ============================================================
def binary_metrics_from_logits(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = labels.numel()
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# Multi-Future Predictor
# ============================================================
class MultiFuturePredictor(nn.Module):
    """
    Conservative multi-future predictor on top of pooled temporal feature z.

    z: [B, D]
    -> shared feature
    -> K residual future branches
    -> branch-wise logits
    """
    def __init__(
        self,
        in_dim=768,
        hidden_dim=512,
        future_dim=256,
        num_branches=3,
        dropout=0.1,
    ):
        super().__init__()

        self.num_branches = num_branches
        self.future_dim = future_dim

        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_dim),
            nn.ReLU(inplace=True),
        )

        self.branch_embeddings = nn.Parameter(torch.randn(num_branches, future_dim) * 0.02)

        self.branch_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(future_dim, future_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(future_dim, future_dim),
            )
            for _ in range(num_branches)
        ])

        self.branch_heads = nn.ModuleList([
            nn.Linear(future_dim, 1)
            for _ in range(num_branches)
        ])

    def forward(self, z):
        # z: [B, D]
        shared_feat = self.shared(z)   # [B, F]

        branch_features = []
        branch_logits = []

        for k in range(self.num_branches):
            base = shared_feat + self.branch_embeddings[k].unsqueeze(0)
            feat = base + self.branch_mlps[k](base)   # residual branch feature
            logit = self.branch_heads[k](feat)        # [B,1]

            branch_features.append(feat)
            branch_logits.append(logit)

        branch_features = torch.stack(branch_features, dim=1)  # [B,K,F]
        branch_logits = torch.stack(branch_logits, dim=1)      # [B,K,1]

        # simple aggregate for decision-level output
        agg_logit = branch_logits.mean(dim=1)                  # [B,1]

        return {
            "shared_feat": shared_feat,
            "branch_features": branch_features,
            "branch_logits": branch_logits,
            "agg_logit": agg_logit,
        }


# ============================================================
# Losses
# ============================================================
def multi_branch_bce_loss(branch_logits, labels):
    """
    BCE across all branches, averaged.
    """
    labels_exp = labels.unsqueeze(1).expand(-1, branch_logits.size(1), -1)
    return F.binary_cross_entropy_with_logits(branch_logits, labels_exp)


def agg_bce_loss(agg_logit, labels):
    return F.binary_cross_entropy_with_logits(agg_logit, labels)


def diversity_loss_cosine(branch_features):
    """
    Encourage branch features to be different.
    """
    B, K, D = branch_features.shape
    if K <= 1:
        return branch_features.new_tensor(0.0)

    x = F.normalize(branch_features, dim=-1)          # [B,K,D]
    sim = torch.matmul(x, x.transpose(1, 2))          # [B,K,K]

    eye = torch.eye(K, device=sim.device).unsqueeze(0)
    offdiag = sim * (1.0 - eye)

    num_pairs = K * K - K
    loss = offdiag.sum() / max(B * num_pairs, 1)
    return loss


def branch_usage_regularizer(branch_logits):
    """
    Encourage all branches to contribute.
    Uses soft positive probability mean across the batch.
    """
    probs = torch.sigmoid(branch_logits).squeeze(-1)    # [B,K]
    usage = probs.mean(dim=0)                           # [K]
    target = torch.full_like(usage, usage.mean().detach())
    return ((usage - target) ** 2).mean(), usage


# ============================================================
# Logging
# ============================================================
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def append_log_csv(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def save_checkpoint(save_path, epoch, predictor, optimizer, train_metrics, val_metrics, args):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
    }
    torch.save(ckpt, save_path)


# ============================================================
# Train / Valid
# ============================================================
def run_one_epoch(
    backbone,
    predictor,
    loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    video_key,
    label_key,
    log_prefix="Train",
    train_mode=True,
    lambda_branch=1.0,
    lambda_agg=1.0,
    lambda_div=0.1,
    lambda_usage=0.1,
    log_interval=50,
):
    if train_mode:
        predictor.train()
    else:
        predictor.eval()

    total_loss_meter = AverageMeter()
    branch_loss_meter = AverageMeter()
    agg_loss_meter = AverageMeter()
    div_loss_meter = AverageMeter()
    usage_loss_meter = AverageMeter()

    all_logits = []
    all_labels = []
    branch_usage_hard = torch.zeros(predictor.num_branches, dtype=torch.long)

    start_time = time.time()
    num_batches = len(loader)

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            video = batch[video_key].to(device, non_blocking=True)
            labels = batch[label_key].float().to(device, non_blocking=True).view(-1, 1)

            with torch.no_grad():
                backbone_out = backbone(video)
                if not isinstance(backbone_out, dict):
                    raise TypeError("TemporalEventModel output must be a dict.")
                if "pooled_feat" not in backbone_out:
                    raise KeyError(f"'pooled_feat' not found in backbone output keys: {list(backbone_out.keys())}")

                z = backbone_out["pooled_feat"]  # [B,768]

            out = predictor(z)
            branch_logits = out["branch_logits"]   # [B,K,1]
            agg_logit = out["agg_logit"]           # [B,1]
            branch_features = out["branch_features"]

            branch_loss = multi_branch_bce_loss(branch_logits, labels)
            agg_loss = agg_bce_loss(agg_logit, labels)
            div_loss = diversity_loss_cosine(branch_features)
            usage_loss, usage_soft = branch_usage_regularizer(branch_logits)

            total_loss = (
                lambda_branch * branch_loss +
                lambda_agg * agg_loss +
                lambda_div * div_loss +
                lambda_usage * usage_loss
            )

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
                optimizer.step()

            bs = video.size(0)

            total_loss_meter.update(total_loss.item(), bs)
            branch_loss_meter.update(branch_loss.item(), bs)
            agg_loss_meter.update(agg_loss.item(), bs)
            div_loss_meter.update(div_loss.item(), bs)
            usage_loss_meter.update(usage_loss.item(), bs)

            all_logits.append(agg_logit.detach().cpu())
            all_labels.append(labels.detach().cpu())

            best_branch = torch.sigmoid(branch_logits).squeeze(-1).argmax(dim=1)
            for idx in best_branch.cpu().tolist():
                branch_usage_hard[idx] += 1

            elapsed = time.time() - start_time
            avg_batch_time = elapsed / batch_idx
            eta = (num_batches - batch_idx) * avg_batch_time

            if (batch_idx % log_interval == 0) or (batch_idx == num_batches):
                batch_metrics = binary_metrics_from_logits(
                    agg_logit.detach().cpu(),
                    labels.detach().cpu()
                )
                usage_str = "[" + ", ".join([f"{u:.3f}" for u in usage_soft.detach().cpu().tolist()]) + "]"

                print(
                    f"\r[{log_prefix}] Epoch {epoch:03d}/{total_epochs:03d} | "
                    f"Batch {batch_idx:04d}/{num_batches:04d} | "
                    f"Loss {total_loss_meter.avg:.4f} | "
                    f"Branch {branch_loss_meter.avg:.4f} | "
                    f"Agg {agg_loss_meter.avg:.4f} | "
                    f"Div {div_loss_meter.avg:.4f} | "
                    f"Usage {usage_loss_meter.avg:.4f} | "
                    f"Acc {batch_metrics['acc']:.4f} | "
                    f"Prec {batch_metrics['precision']:.4f} | "
                    f"Rec {batch_metrics['recall']:.4f} | "
                    f"F1 {batch_metrics['f1']:.4f} | "
                    f"SoftUsage {usage_str} | "
                    f"ETA {format_seconds(eta)}",
                    end=""
                )

    print()

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = binary_metrics_from_logits(all_logits, all_labels)

    metrics["loss"] = total_loss_meter.avg
    metrics["branch_loss"] = branch_loss_meter.avg
    metrics["agg_loss"] = agg_loss_meter.avg
    metrics["div_loss"] = div_loss_meter.avg
    metrics["usage_loss"] = usage_loss_meter.avg
    metrics["branch_usage"] = branch_usage_hard.tolist()

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\temporal_crossing_mamba2scale\best_epoch_002_valF1_0.9670_valAcc_0.9569.pth"
    )

    parser.add_argument(
        "--clips_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD clips"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations"
    )
    parser.add_argument(
        "--attributes_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations_attributes"
    )

    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--early_horizon", type=int, default=30)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--future_dim", type=int, default=256)
    parser.add_argument("--num_branches", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_branch", type=float, default=1.0)
    parser.add_argument("--lambda_agg", type=float, default=1.0)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    parser.add_argument("--lambda_usage", type=float, default=0.1)

    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_decision_mamba2scale"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset_cls = resolve_dataset_class()
    full_dataset = build_dataset(dataset_cls, args)

    sample0 = full_dataset[0]
    video_key, label_key = infer_video_and_label_keys(sample0)

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Dataset too small after split.")

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"[Split] train={len(train_set)} | val={len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------
    backbone = TemporalEventModel()
    ckpt = torch.load(args.backbone_ckpt, map_location="cpu")

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)

    print(f"[Backbone Checkpoint Loaded] {args.backbone_ckpt}")
    print(f"[Backbone Missing Keys] {len(missing)}")
    print(f"[Backbone Unexpected Keys] {len(unexpected)}")
    print()

    backbone = backbone.to(device)
    backbone.eval()

    for p in backbone.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # Multi-Future Predictor
    # --------------------------------------------------------
    predictor = MultiFuturePredictor(
        in_dim=768,
        hidden_dim=args.hidden_dim,
        future_dim=args.future_dim,
        num_branches=args.num_branches,
        dropout=args.dropout,
    ).to(device)

    print_trainable_modules(predictor)
    print(f"[Trainable Params] {count_trainable_params(predictor):,}")

    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    csv_path = os.path.join(args.save_dir, "metrics.csv")
    best_val_f1 = -1.0
    global_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print("\n" + "=" * 110)
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] START")
        print("=" * 110)

        train_metrics = run_one_epoch(
            backbone=backbone,
            predictor=predictor,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            video_key=video_key,
            label_key=label_key,
            log_prefix="Train",
            train_mode=True,
            lambda_branch=args.lambda_branch,
            lambda_agg=args.lambda_agg,
            lambda_div=args.lambda_div,
            lambda_usage=args.lambda_usage,
            log_interval=args.log_interval,
        )

        val_metrics = run_one_epoch(
            backbone=backbone,
            predictor=predictor,
            loader=val_loader,
            optimizer=None,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            video_key=video_key,
            label_key=label_key,
            log_prefix="Valid",
            train_mode=False,
            lambda_branch=args.lambda_branch,
            lambda_agg=args.lambda_agg,
            lambda_div=args.lambda_div,
            lambda_usage=args.lambda_usage,
            log_interval=args.log_interval,
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - global_start
        avg_epoch_time = total_elapsed / epoch
        total_eta = (args.epochs - epoch) * avg_epoch_time

        print("\n" + "-" * 110)
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"Time {format_seconds(epoch_time)} | "
            f"Total Elapsed {format_seconds(total_elapsed)} | "
            f"Remaining {format_seconds(total_eta)}"
        )
        print(
            f"[Train] Loss {train_metrics['loss']:.4f} | "
            f"Branch {train_metrics['branch_loss']:.4f} | "
            f"Agg {train_metrics['agg_loss']:.4f} | "
            f"Div {train_metrics['div_loss']:.4f} | "
            f"Usage {train_metrics['usage_loss']:.4f} | "
            f"Acc {train_metrics['acc']:.4f} | "
            f"Prec {train_metrics['precision']:.4f} | "
            f"Rec {train_metrics['recall']:.4f} | "
            f"F1 {train_metrics['f1']:.4f} | "
            f"BranchUsage {train_metrics['branch_usage']}"
        )
        print(
            f"[Valid] Loss {val_metrics['loss']:.4f} | "
            f"Branch {val_metrics['branch_loss']:.4f} | "
            f"Agg {val_metrics['agg_loss']:.4f} | "
            f"Div {val_metrics['div_loss']:.4f} | "
            f"Usage {val_metrics['usage_loss']:.4f} | "
            f"Acc {val_metrics['acc']:.4f} | "
            f"Prec {val_metrics['precision']:.4f} | "
            f"Rec {val_metrics['recall']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"BranchUsage {val_metrics['branch_usage']}"
        )
        print("-" * 110)

        epoch_ckpt_path = os.path.join(
            args.save_dir,
            f"epoch_{epoch:03d}_valF1_{val_metrics['f1']:.4f}_valAcc_{val_metrics['acc']:.4f}.pth"
        )
        save_checkpoint(epoch_ckpt_path, epoch, predictor, optimizer, train_metrics, val_metrics, args)
        print(f"[Checkpoint Saved] {epoch_ckpt_path}")

        latest_ckpt_path = os.path.join(args.save_dir, "latest.pth")
        save_checkpoint(latest_ckpt_path, epoch, predictor, optimizer, train_metrics, val_metrics, args)
        print(f"[Latest Updated] {latest_ckpt_path}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_ckpt_path = os.path.join(
                args.save_dir,
                f"best_epoch_{epoch:03d}_valF1_{val_metrics['f1']:.4f}_valAcc_{val_metrics['acc']:.4f}.pth"
            )
            save_checkpoint(best_ckpt_path, epoch, predictor, optimizer, train_metrics, val_metrics, args)
            print(f"[Best Updated] {best_ckpt_path}")

        row = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 4),
            "total_elapsed_sec": round(total_elapsed, 4),

            "train_loss": round(train_metrics["loss"], 6),
            "train_branch_loss": round(train_metrics["branch_loss"], 6),
            "train_agg_loss": round(train_metrics["agg_loss"], 6),
            "train_div_loss": round(train_metrics["div_loss"], 6),
            "train_usage_loss": round(train_metrics["usage_loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "train_precision": round(train_metrics["precision"], 6),
            "train_recall": round(train_metrics["recall"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "train_branch_usage": str(train_metrics["branch_usage"]),

            "val_loss": round(val_metrics["loss"], 6),
            "val_branch_loss": round(val_metrics["branch_loss"], 6),
            "val_agg_loss": round(val_metrics["agg_loss"], 6),
            "val_div_loss": round(val_metrics["div_loss"], 6),
            "val_usage_loss": round(val_metrics["usage_loss"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_branch_usage": str(val_metrics["branch_usage"]),
        }
        append_log_csv(csv_path, row)
        print(f"[Metrics Logged] {csv_path}")

    print("\n" + "=" * 110)
    print(f"[Training Done] Best Val F1 = {best_val_f1:.4f}")
    print("=" * 110)


if __name__ == "__main__":
    main()