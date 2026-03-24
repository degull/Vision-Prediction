# C:\Users\IIPL02\Desktop\Vision Prediction\train_multi_future_stage2.py
import os
import csv
import time
import random
import argparse
import inspect
import importlib
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from models.multi_future_stage2 import (
    ContextExpertStage2Model,
    compute_stage2_losses,
)


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
            print(f"{name:100s} {tuple(p.shape)}")
    print()


def build_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


# ============================================================
# Dataset Resolver
# ============================================================
def resolve_dataset_class():
    candidates = [
        ("datasets.jaad_crossing_clip_context_dataset", ["JAADCrossingClipContextDataset"]),
        ("jaad_crossing_clip_context_dataset", ["JAADCrossingClipContextDataset"]),
        ("datasets.jaad_video_dataset", ["JAADVideoDataset"]),   # fallback
        ("jaad_video_dataset", ["JAADVideoDataset"]),            # fallback
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

    raise ImportError(
        "Could not resolve JAAD dataset class.\n" + "\n".join(errors)
    )


def build_dataset(dataset_cls, args):
    sig = inspect.signature(dataset_cls.__init__)
    supported = set(sig.parameters.keys())

    candidate_kwargs = {
        "clips_dir": args.clips_dir,
        "annotations_dir": args.annotations_dir,
        "attributes_dir": args.attributes_dir,
        "appearance_dir": args.appearance_dir,
        "traffic_dir": args.traffic_dir,
        "vehicle_dir": args.vehicle_dir,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "frame_stride": args.frame_stride,
        "sample_stride": args.sample_stride,
        "early_horizon": args.early_horizon,
        "verbose": args.dataset_verbose,
    }

    final_kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}

    print("[Dataset Init Args]")
    for k, v in final_kwargs.items():
        print(f"  {k}: {v}")
    print()

    return dataset_cls(**final_kwargs)


def infer_keys(sample):
    if not isinstance(sample, dict):
        raise TypeError("Dataset sample must be dict.")

    required_keys = [
        "video",
        "crossing_label",
        "attr_vec",
        "app_vec",
        "traffic_vec",
        "vehicle_vec",
    ]
    for k in required_keys:
        if k not in sample:
            raise KeyError(f"Required key '{k}' not found in sample keys: {list(sample.keys())}")

    print("[Detected Sample Keys]")
    for k in sample.keys():
        print(f"  {k}")
    print()

    return {
        "video_key": "video",
        "label_key": "crossing_label",
        "attr_key": "attr_vec",
        "app_key": "app_vec",
        "traffic_key": "traffic_vec",
        "vehicle_key": "vehicle_vec",
    }


def get_subset_label_stats(subset, label_key):
    pos, neg = 0, 0
    for i in range(len(subset)):
        sample = subset[i]
        y = sample[label_key]
        if isinstance(y, torch.Tensor):
            y = y.item()
        y = int(y)
        if y == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def build_weighted_sampler(subset, label_key):
    labels = []
    for i in range(len(subset)):
        sample = subset[i]
        y = sample[label_key]
        if isinstance(y, torch.Tensor):
            y = y.item()
        labels.append(int(y))

    num_neg = sum(1 for y in labels if y == 0)
    num_pos = sum(1 for y in labels if y == 1)

    class_weights = {
        0: 1.0 / max(num_neg, 1),
        1: 1.0 / max(num_pos, 1),
    }

    sample_weights = [class_weights[y] for y in labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def compute_dataset_pos_weight(pos_count, neg_count, max_pos_weight=10.0):
    pos_weight = float(neg_count) / max(float(pos_count), 1.0)
    pos_weight = max(1e-6, min(pos_weight, max_pos_weight))
    return pos_weight


# ============================================================
# Metrics
# ============================================================
def binary_metrics_from_logits(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = labels.numel()
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(logits, labels):
    best_thr = 0.5
    best_score = -1.0
    best_metrics = None

    for thr in [i / 100 for i in range(10, 91, 5)]:
        m = binary_metrics_from_logits(logits, labels, threshold=thr)
        score = m["balanced_acc"]
        if score > best_score:
            best_score = score
            best_thr = thr
            best_metrics = m

    return best_thr, best_metrics


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


def save_checkpoint(save_path, epoch, model, optimizer, train_metrics, val_metrics, args):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
    }
    torch.save(ckpt, save_path)


# ============================================================
# Stage1 checkpoint loader
# ============================================================
def load_stage1_checkpoint(stage2_model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = stage2_model.stage1_model.load_state_dict(state_dict, strict=False)

    print(f"[Stage1 Checkpoint Loaded] {ckpt_path}")
    print(f"[Stage1 Missing Keys] {len(missing)}")
    print(f"[Stage1 Unexpected Keys] {len(unexpected)}")
    if len(missing) > 0:
        print("  Missing examples:", missing[:10])
    if len(unexpected) > 0:
        print("  Unexpected examples:", unexpected[:10])
    print()


# ============================================================
# Train / Valid
# ============================================================
def run_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    keys,
    threshold,
    pos_weight,
    log_prefix="Train",
    train_mode=True,
    lambda_final=1.0,
    lambda_branch=0.3,
    lambda_distill=0.3,
    lambda_gate_balance=0.05,
    lambda_div=0.10,
    lambda_margin=0.05,
    lambda_risk_reg=0.01,
    log_interval=50,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss_meter = AverageMeter()
    final_loss_meter = AverageMeter()
    branch_loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter()
    gate_balance_meter = AverageMeter()
    div_meter = AverageMeter()
    margin_meter = AverageMeter()
    risk_reg_meter = AverageMeter()

    all_logits = []
    all_labels = []
    gate_usage_sum = torch.zeros(2, dtype=torch.float64)
    gate_usage_count = 0
    branch_usage_hard = torch.zeros(2, dtype=torch.long)

    start_time = time.time()
    num_batches = len(loader)

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            video = batch[keys["video_key"]].to(device, non_blocking=True)
            labels = batch[keys["label_key"]].float().to(device, non_blocking=True).view(-1, 1)

            attr_vec = batch[keys["attr_key"]].float().to(device, non_blocking=True)
            app_vec = batch[keys["app_key"]].float().to(device, non_blocking=True)
            traffic_vec = batch[keys["traffic_key"]].float().to(device, non_blocking=True)
            vehicle_vec = batch[keys["vehicle_key"]].float().to(device, non_blocking=True)

            outputs = model(video, attr_vec, app_vec, traffic_vec, vehicle_vec)
            losses = compute_stage2_losses(
                outputs=outputs,
                labels=labels,
                pos_weight=pos_weight,
                sample_weight=None,
                lambda_final=lambda_final,
                lambda_branch=lambda_branch,
                lambda_distill=lambda_distill,
                lambda_gate_balance=lambda_gate_balance,
                lambda_div=lambda_div,
                lambda_margin=lambda_margin,
                lambda_risk_reg=lambda_risk_reg,
            )

            total_loss = losses["total"]

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            final_logit = outputs["final_logit"]
            gate_weights = outputs["gate_weights"]

            bs = video.size(0)

            total_loss_meter.update(total_loss.item(), bs)
            final_loss_meter.update(losses["final"].item(), bs)
            branch_loss_meter.update(losses["branch"].item(), bs)
            distill_loss_meter.update(losses["distill"].item(), bs)
            gate_balance_meter.update(losses["gate_balance"].item(), bs)
            div_meter.update(losses["div"].item(), bs)
            margin_meter.update(losses["margin"].item(), bs)
            risk_reg_meter.update(losses["risk_reg"].item(), bs)

            all_logits.append(final_logit.detach().cpu())
            all_labels.append(labels.detach().cpu())

            gate_usage_sum += gate_weights.detach().cpu().sum(dim=0).double()
            gate_usage_count += gate_weights.size(0)

            best_branch = gate_weights.detach().cpu().argmax(dim=1)
            for idx in best_branch.tolist():
                branch_usage_hard[idx] += 1

            elapsed = time.time() - start_time
            avg_batch_time = elapsed / batch_idx
            eta = (num_batches - batch_idx) * avg_batch_time

            if (batch_idx % log_interval == 0) or (batch_idx == num_batches):
                running_logits = torch.cat(all_logits, dim=0)
                running_labels = torch.cat(all_labels, dim=0)
                batch_metrics = binary_metrics_from_logits(
                    running_logits,
                    running_labels,
                    threshold=threshold,
                )

                mean_gate = (gate_usage_sum / max(gate_usage_count, 1)).tolist()
                gate_str = "[" + ", ".join([f"{g:.3f}" for g in mean_gate]) + "]"

                print(
                    f"\r[{log_prefix}] Epoch {epoch:03d}/{total_epochs:03d} | "
                    f"Batch {batch_idx:04d}/{num_batches:04d} | "
                    f"Loss {total_loss_meter.avg:.4f} | "
                    f"Final {final_loss_meter.avg:.4f} | "
                    f"Branch {branch_loss_meter.avg:.4f} | "
                    f"Distill {distill_loss_meter.avg:.4f} | "
                    f"GateBal {gate_balance_meter.avg:.4f} | "
                    f"Div {div_meter.avg:.4f} | "
                    f"Margin {margin_meter.avg:.4f} | "
                    f"RiskReg {risk_reg_meter.avg:.4f} | "
                    f"Acc {batch_metrics['acc']:.4f} | "
                    f"BalAcc {batch_metrics['balanced_acc']:.4f} | "
                    f"Prec {batch_metrics['precision']:.4f} | "
                    f"Rec {batch_metrics['recall']:.4f} | "
                    f"Spec {batch_metrics['specificity']:.4f} | "
                    f"F1 {batch_metrics['f1']:.4f} | "
                    f"TP {batch_metrics['tp']} | "
                    f"TN {batch_metrics['tn']} | "
                    f"FP {batch_metrics['fp']} | "
                    f"FN {batch_metrics['fn']} | "
                    f"Gate {gate_str} | "
                    f"ETA {format_seconds(eta)}",
                    end=""
                )

    print()

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = binary_metrics_from_logits(all_logits, all_labels, threshold=threshold)
    best_thr, best_thr_metrics = find_best_threshold(all_logits, all_labels)

    metrics["loss"] = total_loss_meter.avg
    metrics["final_loss"] = final_loss_meter.avg
    metrics["branch_loss"] = branch_loss_meter.avg
    metrics["distill_loss"] = distill_loss_meter.avg
    metrics["gate_balance_loss"] = gate_balance_meter.avg
    metrics["div_loss"] = div_meter.avg
    metrics["margin_loss"] = margin_meter.avg
    metrics["risk_reg_loss"] = risk_reg_meter.avg
    metrics["gate_mean"] = (gate_usage_sum / max(gate_usage_count, 1)).tolist()
    metrics["branch_usage"] = branch_usage_hard.tolist()

    metrics["best_threshold"] = best_thr
    metrics["best_thr_acc"] = best_thr_metrics["acc"]
    metrics["best_thr_precision"] = best_thr_metrics["precision"]
    metrics["best_thr_recall"] = best_thr_metrics["recall"]
    metrics["best_thr_specificity"] = best_thr_metrics["specificity"]
    metrics["best_thr_balanced_acc"] = best_thr_metrics["balanced_acc"]
    metrics["best_thr_f1"] = best_thr_metrics["f1"]
    metrics["best_thr_tp"] = best_thr_metrics["tp"]
    metrics["best_thr_tn"] = best_thr_metrics["tn"]
    metrics["best_thr_fp"] = best_thr_metrics["fp"]
    metrics["best_thr_fn"] = best_thr_metrics["fn"]

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage1_ckpt",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\temporal_crossing\best_epoch_005_valF1_0.9777_valAcc_0.9706.pth"
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
    parser.add_argument(
        "--appearance_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations_appearance"
    )
    parser.add_argument(
        "--traffic_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations_traffic"
    )
    parser.add_argument(
        "--vehicle_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations_vehicle"
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

    parser.add_argument("--stage1_feat_dim", type=int, default=768)

    parser.add_argument("--attr_dim", type=int, default=6)
    parser.add_argument("--app_dim", type=int, default=5)
    parser.add_argument("--traffic_dim", type=int, default=6)
    parser.add_argument("--vehicle_dim", type=int, default=6)

    parser.add_argument("--context_embed_dim", type=int, default=64)
    parser.add_argument("--context_hidden_dim", type=int, default=64)
    parser.add_argument("--expert_hidden_dim", type=int, default=128)
    parser.add_argument("--gate_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--base_logit_weight", type=float, default=0.6)
    parser.add_argument("--expert_logit_weight", type=float, default=0.4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_unfreeze", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_final", type=float, default=1.0)
    parser.add_argument("--lambda_branch", type=float, default=0.3)
    parser.add_argument("--lambda_distill", type=float, default=0.3)
    parser.add_argument("--lambda_gate_balance", type=float, default=0.05)
    parser.add_argument("--lambda_div", type=float, default=0.10)
    parser.add_argument("--lambda_margin", type=float, default=0.05)
    parser.add_argument("--lambda_risk_reg", type=float, default=0.01)

    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--dataset_verbose", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--unfreeze_epoch", type=int, default=100)
    parser.add_argument("--max_pos_weight", type=float, default=10.0)

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\context_expert_stage2_2branch"
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
    keys = infer_keys(sample0)

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

    train_pos, train_neg = get_subset_label_stats(train_set, keys["label_key"])
    val_pos, val_neg = get_subset_label_stats(val_set, keys["label_key"])
    print(f"[Train Labels] pos={train_pos}, neg={train_neg}")
    print(f"[Valid Labels] pos={val_pos}, neg={val_neg}")

    dataset_pos_weight = compute_dataset_pos_weight(
        train_pos,
        train_neg,
        max_pos_weight=args.max_pos_weight,
    )
    print(f"[Loss] dataset_pos_weight={dataset_pos_weight:.4f}")

    train_sampler = build_weighted_sampler(train_set, keys["label_key"])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
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
    # Model
    # --------------------------------------------------------
    model = ContextExpertStage2Model(
        stage1_model=None,
        stage1_feat_dim=args.stage1_feat_dim,

        attr_dim=args.attr_dim,
        app_dim=args.app_dim,
        traffic_dim=args.traffic_dim,
        vehicle_dim=args.vehicle_dim,

        context_embed_dim=args.context_embed_dim,
        context_hidden_dim=args.context_hidden_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        dropout=args.dropout,
        base_logit_weight=args.base_logit_weight,
        expert_logit_weight=args.expert_logit_weight,

        # -----------------------------
        # Stage1 config (must match Stage1 training)
        # -----------------------------
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

    load_stage1_checkpoint(model, args.stage1_ckpt)

    # Freeze Stage1 by default
    model.freeze_stage1()

    print_trainable_modules(model)
    print(f"[Trainable Params] {count_trainable_params(model):,}")

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    csv_path = os.path.join(args.save_dir, "metrics.csv")
    best_val_bal_acc = -1.0
    global_start = time.time()
    has_unfroze = False

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if (not has_unfroze) and (epoch >= args.unfreeze_epoch):
            print("\n" + "=" * 120)
            print(f"[Stage1 Partial Unfreeze @ Epoch {epoch}]")
            print("=" * 120)

            model.unfreeze_stage1_temporal_and_head_only()
            print_trainable_modules(model)
            print(f"[Trainable Params After Unfreeze] {count_trainable_params(model):,}")

            optimizer = build_optimizer(model, lr=args.lr_unfreeze, weight_decay=args.weight_decay)
            has_unfroze = True

        print("\n" + "=" * 120)
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] START")
        print("=" * 120)

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            keys=keys,
            threshold=args.decision_threshold,
            pos_weight=dataset_pos_weight,
            log_prefix="Train",
            train_mode=True,
            lambda_final=args.lambda_final,
            lambda_branch=args.lambda_branch,
            lambda_distill=args.lambda_distill,
            lambda_gate_balance=args.lambda_gate_balance,
            lambda_div=args.lambda_div,
            lambda_margin=args.lambda_margin,
            lambda_risk_reg=args.lambda_risk_reg,
            log_interval=args.log_interval,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            keys=keys,
            threshold=args.decision_threshold,
            pos_weight=dataset_pos_weight,
            log_prefix="Valid",
            train_mode=False,
            lambda_final=args.lambda_final,
            lambda_branch=args.lambda_branch,
            lambda_distill=args.lambda_distill,
            lambda_gate_balance=args.lambda_gate_balance,
            lambda_div=args.lambda_div,
            lambda_margin=args.lambda_margin,
            lambda_risk_reg=args.lambda_risk_reg,
            log_interval=args.log_interval,
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - global_start
        avg_epoch_time = total_elapsed / epoch
        total_eta = (args.epochs - epoch) * avg_epoch_time

        print("\n" + "-" * 120)
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"Time {format_seconds(epoch_time)} | "
            f"Total Elapsed {format_seconds(total_elapsed)} | "
            f"Remaining {format_seconds(total_eta)}"
        )
        print(
            f"[Train] Loss {train_metrics['loss']:.4f} | "
            f"Final {train_metrics['final_loss']:.4f} | "
            f"Branch {train_metrics['branch_loss']:.4f} | "
            f"Distill {train_metrics['distill_loss']:.4f} | "
            f"GateBal {train_metrics['gate_balance_loss']:.4f} | "
            f"Div {train_metrics['div_loss']:.4f} | "
            f"Margin {train_metrics['margin_loss']:.4f} | "
            f"RiskReg {train_metrics['risk_reg_loss']:.4f} | "
            f"Acc {train_metrics['acc']:.4f} | "
            f"BalAcc {train_metrics['balanced_acc']:.4f} | "
            f"Prec {train_metrics['precision']:.4f} | "
            f"Rec {train_metrics['recall']:.4f} | "
            f"Spec {train_metrics['specificity']:.4f} | "
            f"F1 {train_metrics['f1']:.4f} | "
            f"TP {train_metrics['tp']} | TN {train_metrics['tn']} | "
            f"FP {train_metrics['fp']} | FN {train_metrics['fn']} | "
            f"BestThr {train_metrics['best_threshold']:.2f} | "
            f"BestThrBalAcc {train_metrics['best_thr_balanced_acc']:.4f} | "
            f"GateMean {train_metrics['gate_mean']} | "
            f"BranchUsage {train_metrics['branch_usage']}"
        )
        print(
            f"[Valid] Loss {val_metrics['loss']:.4f} | "
            f"Final {val_metrics['final_loss']:.4f} | "
            f"Branch {val_metrics['branch_loss']:.4f} | "
            f"Distill {val_metrics['distill_loss']:.4f} | "
            f"GateBal {val_metrics['gate_balance_loss']:.4f} | "
            f"Div {val_metrics['div_loss']:.4f} | "
            f"Margin {val_metrics['margin_loss']:.4f} | "
            f"RiskReg {val_metrics['risk_reg_loss']:.4f} | "
            f"Acc {val_metrics['acc']:.4f} | "
            f"BalAcc {val_metrics['balanced_acc']:.4f} | "
            f"Prec {val_metrics['precision']:.4f} | "
            f"Rec {val_metrics['recall']:.4f} | "
            f"Spec {val_metrics['specificity']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"TP {val_metrics['tp']} | TN {val_metrics['tn']} | "
            f"FP {val_metrics['fp']} | FN {val_metrics['fn']} | "
            f"BestThr {val_metrics['best_threshold']:.2f} | "
            f"BestThrBalAcc {val_metrics['best_thr_balanced_acc']:.4f} | "
            f"GateMean {val_metrics['gate_mean']} | "
            f"BranchUsage {val_metrics['branch_usage']}"
        )
        print("-" * 120)

        epoch_ckpt_path = os.path.join(
            args.save_dir,
            f"epoch_{epoch:03d}_valBalAcc_{val_metrics['balanced_acc']:.4f}_valF1_{val_metrics['f1']:.4f}.pth"
        )
        save_checkpoint(epoch_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
        print(f"[Checkpoint Saved] {epoch_ckpt_path}")

        latest_ckpt_path = os.path.join(args.save_dir, "latest.pth")
        save_checkpoint(latest_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
        print(f"[Latest Updated] {latest_ckpt_path}")

        if val_metrics["best_thr_balanced_acc"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["best_thr_balanced_acc"]
            best_ckpt_path = os.path.join(
                args.save_dir,
                f"best_epoch_{epoch:03d}_valBestThrBalAcc_{val_metrics['best_thr_balanced_acc']:.4f}_valF1_{val_metrics['best_thr_f1']:.4f}.pth"
            )
            save_checkpoint(best_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
            print(f"[Best Updated] {best_ckpt_path}")

        row = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 4),
            "total_elapsed_sec": round(total_elapsed, 4),

            "train_loss": round(train_metrics["loss"], 6),
            "train_final_loss": round(train_metrics["final_loss"], 6),
            "train_branch_loss": round(train_metrics["branch_loss"], 6),
            "train_distill_loss": round(train_metrics["distill_loss"], 6),
            "train_gate_balance_loss": round(train_metrics["gate_balance_loss"], 6),
            "train_div_loss": round(train_metrics["div_loss"], 6),
            "train_margin_loss": round(train_metrics["margin_loss"], 6),
            "train_risk_reg_loss": round(train_metrics["risk_reg_loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "train_balanced_acc": round(train_metrics["balanced_acc"], 6),
            "train_precision": round(train_metrics["precision"], 6),
            "train_recall": round(train_metrics["recall"], 6),
            "train_specificity": round(train_metrics["specificity"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "train_tp": train_metrics["tp"],
            "train_tn": train_metrics["tn"],
            "train_fp": train_metrics["fp"],
            "train_fn": train_metrics["fn"],
            "train_best_threshold": round(train_metrics["best_threshold"], 4),
            "train_best_thr_balanced_acc": round(train_metrics["best_thr_balanced_acc"], 6),
            "train_best_thr_f1": round(train_metrics["best_thr_f1"], 6),
            "train_gate_mean": str(train_metrics["gate_mean"]),
            "train_branch_usage": str(train_metrics["branch_usage"]),

            "val_loss": round(val_metrics["loss"], 6),
            "val_final_loss": round(val_metrics["final_loss"], 6),
            "val_branch_loss": round(val_metrics["branch_loss"], 6),
            "val_distill_loss": round(val_metrics["distill_loss"], 6),
            "val_gate_balance_loss": round(val_metrics["gate_balance_loss"], 6),
            "val_div_loss": round(val_metrics["div_loss"], 6),
            "val_margin_loss": round(val_metrics["margin_loss"], 6),
            "val_risk_reg_loss": round(val_metrics["risk_reg_loss"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "val_balanced_acc": round(val_metrics["balanced_acc"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_specificity": round(val_metrics["specificity"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_tp": val_metrics["tp"],
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_best_threshold": round(val_metrics["best_threshold"], 4),
            "val_best_thr_balanced_acc": round(val_metrics["best_thr_balanced_acc"], 6),
            "val_best_thr_f1": round(val_metrics["best_thr_f1"], 6),
            "val_gate_mean": str(val_metrics["gate_mean"]),
            "val_branch_usage": str(val_metrics["branch_usage"]),
        }
        append_log_csv(csv_path, row)
        print(f"[Metrics Logged] {csv_path}")

    print("\n" + "=" * 120)
    print(f"[Training Done] Best Val Best-Threshold Balanced Acc = {best_val_bal_acc:.4f}")
    print("=" * 120)


if __name__ == "__main__":
    main()