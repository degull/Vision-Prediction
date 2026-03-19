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
    MultiFutureStage2Model,
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


# ============================================================
# Dataset Resolver
# ============================================================
def resolve_dataset_class():
    candidates = [
        ("datasets.jaad_video_dataset", ["JAADVideoDataset"]),
        ("jaad_video_dataset", ["JAADVideoDataset"]),
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


def infer_video_and_label_keys(sample):
    if not isinstance(sample, dict):
        raise TypeError("Dataset sample must be dict.")

    video_key_candidates = ["video", "frames", "clip", "images"]
    label_key_candidates = [
        "crossing_label",
        "crossing",
        "label",
        "target",
        "y",
        "event_label",
        "crossing_target",
        "pedestrian_crossing",
    ]

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
        "optimizer_state_dict": optimizer.state_dict(),
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
    video_key,
    label_key,
    threshold,
    log_prefix="Train",
    train_mode=True,
    lambda_final=1.0,
    lambda_branch=0.05,
    lambda_balance=0.01,
    lambda_div=0.005,
    lambda_distill=1.0,
    log_interval=50,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss_meter = AverageMeter()
    final_loss_meter = AverageMeter()
    branch_loss_meter = AverageMeter()
    balance_loss_meter = AverageMeter()
    div_loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter()

    all_logits = []
    all_labels = []
    branch_usage_hard = torch.zeros(model.num_branches, dtype=torch.long)

    start_time = time.time()
    num_batches = len(loader)

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            video = batch[video_key].to(device, non_blocking=True)
            labels = batch[label_key].float().to(device, non_blocking=True).view(-1, 1)

            outputs = model(video)
            losses = compute_stage2_losses(
                outputs=outputs,
                labels=labels,
                lambda_final=lambda_final,
                lambda_branch=lambda_branch,
                lambda_balance=lambda_balance,
                lambda_div=lambda_div,
                lambda_distill=lambda_distill,
            )

            total_loss = losses["total"]

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            final_logit = outputs["final_logit"]
            branch_conf = outputs["branch_conf"]

            bs = video.size(0)

            total_loss_meter.update(total_loss.item(), bs)
            final_loss_meter.update(losses["final"].item(), bs)
            branch_loss_meter.update(losses["branch"].item(), bs)
            balance_loss_meter.update(losses["balance"].item(), bs)
            div_loss_meter.update(losses["div"].item(), bs)
            distill_loss_meter.update(losses["distill"].item(), bs)

            all_logits.append(final_logit.detach().cpu())
            all_labels.append(labels.detach().cpu())

            best_branch = branch_conf.detach().cpu().argmax(dim=1)
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

                usage_mean = branch_conf.detach().cpu().mean(dim=0).tolist()
                usage_str = "[" + ", ".join([f"{u:.3f}" for u in usage_mean]) + "]"

                print(
                    f"\r[{log_prefix}] Epoch {epoch:03d}/{total_epochs:03d} | "
                    f"Batch {batch_idx:04d}/{num_batches:04d} | "
                    f"Loss {total_loss_meter.avg:.4f} | "
                    f"Final {final_loss_meter.avg:.4f} | "
                    f"Branch {branch_loss_meter.avg:.4f} | "
                    f"Balance {balance_loss_meter.avg:.4f} | "
                    f"Div {div_loss_meter.avg:.4f} | "
                    f"Distill {distill_loss_meter.avg:.4f} | "
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
                    f"BranchConf {usage_str} | "
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
    metrics["balance_loss"] = balance_loss_meter.avg
    metrics["div_loss"] = div_loss_meter.avg
    metrics["distill_loss"] = distill_loss_meter.avg
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

    parser.add_argument("--stage1_feat_dim", type=int, default=768)
    parser.add_argument("--adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--future_dim", type=int, default=128)
    parser.add_argument("--shared_hidden_dim", type=int, default=128)
    parser.add_argument("--branch_hidden_dim", type=int, default=128)
    parser.add_argument("--num_branches", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base_logit_weight", type=float, default=0.85)
    parser.add_argument("--future_logit_weight", type=float, default=0.15)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_final", type=float, default=1.0)
    parser.add_argument("--lambda_branch", type=float, default=0.05)
    parser.add_argument("--lambda_balance", type=float, default=0.01)
    parser.add_argument("--lambda_div", type=float, default=0.005)
    parser.add_argument("--lambda_distill", type=float, default=1.0)

    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--dataset_verbose", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_stage2_basepreserve"
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
    print(f"[Sample Keys] {list(sample0.keys())}")
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

    train_pos, train_neg = get_subset_label_stats(train_set, label_key)
    val_pos, val_neg = get_subset_label_stats(val_set, label_key)
    print(f"[Train Labels] pos={train_pos}, neg={train_neg}")
    print(f"[Valid Labels] pos={val_pos}, neg={val_neg}")

    train_sampler = build_weighted_sampler(train_set, label_key)

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
    model = MultiFutureStage2Model(
        stage1_model=None,
        stage1_feat_dim=args.stage1_feat_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        future_dim=args.future_dim,
        shared_hidden_dim=args.shared_hidden_dim,
        branch_hidden_dim=args.branch_hidden_dim,
        num_branches=args.num_branches,
        dropout=args.dropout,
        base_logit_weight=args.base_logit_weight,
        future_logit_weight=args.future_logit_weight,
    ).to(device)

    load_stage1_checkpoint(model, args.stage1_ckpt)

    # Warm-up: freeze full stage1
    model.freeze_stage1()

    print_trainable_modules(model)
    print(f"[Trainable Params] {count_trainable_params(model):,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    csv_path = os.path.join(args.save_dir, "metrics.csv")
    best_val_bal_acc = -1.0
    global_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

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
            video_key=video_key,
            label_key=label_key,
            threshold=args.decision_threshold,
            log_prefix="Train",
            train_mode=True,
            lambda_final=args.lambda_final,
            lambda_branch=args.lambda_branch,
            lambda_balance=args.lambda_balance,
            lambda_div=args.lambda_div,
            lambda_distill=args.lambda_distill,
            log_interval=args.log_interval,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            video_key=video_key,
            label_key=label_key,
            threshold=args.decision_threshold,
            log_prefix="Valid",
            train_mode=False,
            lambda_final=args.lambda_final,
            lambda_branch=args.lambda_branch,
            lambda_balance=args.lambda_balance,
            lambda_div=args.lambda_div,
            lambda_distill=args.lambda_distill,
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
            f"Balance {train_metrics['balance_loss']:.4f} | "
            f"Div {train_metrics['div_loss']:.4f} | "
            f"Distill {train_metrics['distill_loss']:.4f} | "
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
            f"BranchUsage {train_metrics['branch_usage']}"
        )
        print(
            f"[Valid] Loss {val_metrics['loss']:.4f} | "
            f"Final {val_metrics['final_loss']:.4f} | "
            f"Branch {val_metrics['branch_loss']:.4f} | "
            f"Balance {val_metrics['balance_loss']:.4f} | "
            f"Div {val_metrics['div_loss']:.4f} | "
            f"Distill {val_metrics['distill_loss']:.4f} | "
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
            "train_balance_loss": round(train_metrics["balance_loss"], 6),
            "train_div_loss": round(train_metrics["div_loss"], 6),
            "train_distill_loss": round(train_metrics["distill_loss"], 6),
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
            "train_branch_usage": str(train_metrics["branch_usage"]),

            "val_loss": round(val_metrics["loss"], 6),
            "val_final_loss": round(val_metrics["final_loss"], 6),
            "val_branch_loss": round(val_metrics["branch_loss"], 6),
            "val_balance_loss": round(val_metrics["balance_loss"], 6),
            "val_div_loss": round(val_metrics["div_loss"], 6),
            "val_distill_loss": round(val_metrics["distill_loss"], 6),
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
            "val_branch_usage": str(val_metrics["branch_usage"]),
        }
        append_log_csv(csv_path, row)
        print(f"[Metrics Logged] {csv_path}")

    print("\n" + "=" * 120)
    print(f"[Training Done] Best Val Best-Threshold Balanced Acc = {best_val_bal_acc:.4f}")
    print("=" * 120)


if __name__ == "__main__":
    main()