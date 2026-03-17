# 3. Multi-Future Predictor 단계
import os
import csv
import cv2
import time
import random
import argparse
import xml.etree.ElementTree as ET
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from models.future_model import MultiFutureCrossingModel


# ============================================================
# Utility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_modules(model):
    print("\n[Trainable Parameters]")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:65s} {tuple(p.shape)}")
    print()


# ============================================================
# Metrics
# ============================================================
def binary_metrics_from_probs(probs: torch.Tensor, labels: torch.Tensor):
    """
    probs : [B, 1]
    labels: [B, 1]
    """
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
# Exact JAAD parser (same target as stage-1)
# ============================================================
class JAADCrossingParser:
    """
    Parse exact JAAD pedestrian annotation XML from:
        JAAD annotations/annotations/video_xxxx.xml

    Label rule for each frame:
        - if any visible pedestrian has cross == "crossing" -> label 1
        - else if at least one visible pedestrian has cross == "not-crossing" -> label 0
        - else -> unlabeled / skipped
    """

    def __init__(self):
        self.pos_label = "crossing"
        self.neg_label = "not-crossing"

    def parse_frame_labels(self, xml_path: str):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        frame_cross_values = {}

        tracks = root.findall(".//track")

        for track in tracks:
            label_name = track.attrib.get("label", "").strip().lower()
            if label_name != "pedestrian":
                continue

            for box in track.findall("box"):
                outside = box.attrib.get("outside", "0").strip()
                if outside == "1":
                    continue

                frame_id = int(box.attrib["frame"])

                attrs = {}
                for attr in box.findall("attribute"):
                    attr_name = attr.attrib.get("name", "").strip().lower()
                    attr_value = (attr.text or "").strip().lower()
                    attrs[attr_name] = attr_value

                cross_value = attrs.get("cross", "")
                if cross_value == "":
                    continue

                frame_cross_values.setdefault(frame_id, []).append(cross_value)

        frame_labels = {}
        for frame_id, values in frame_cross_values.items():
            if any(v == self.pos_label for v in values):
                frame_labels[frame_id] = 1.0
            elif any(v == self.neg_label for v in values):
                frame_labels[frame_id] = 0.0

        return frame_labels


# ============================================================
# Dataset
# ============================================================
class JAADCrossingClipDataset(Dataset):
    """
    Sliding clip dataset:
        frames [t-(T-1)*frame_stride, ..., t]

    Label:
        crossing label at last frame t
    """

    def __init__(
        self,
        clips_dir: str,
        annotations_dir: str,
        num_frames: int = 8,
        image_size: int = 224,
        frame_stride: int = 1,
        sample_stride: int = 8,
        verbose: bool = True,
    ):
        super().__init__()

        self.clips_dir = clips_dir
        self.annotations_dir = annotations_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.sample_stride = sample_stride

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.parser = JAADCrossingParser()
        self.samples = []

        video_files = sorted([
            f for f in os.listdir(clips_dir)
            if f.lower().endswith(".mp4")
        ])

        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(clips_dir, video_file)
            xml_path = os.path.join(annotations_dir, f"{video_name}.xml")

            if not os.path.exists(xml_path):
                continue

            try:
                frame_labels = self.parser.parse_frame_labels(xml_path)
            except Exception as e:
                print(f"[WARN] Failed to parse XML: {xml_path} | {e}")
                continue

            if len(frame_labels) == 0:
                continue

            valid_end_frames = sorted(frame_labels.keys())

            for end_frame in valid_end_frames[::sample_stride]:
                start_frame = end_frame - (num_frames - 1) * frame_stride
                if start_frame < 0:
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "xml_path": xml_path,
                    "end_frame": end_frame,
                    "label": frame_labels[end_frame],
                })

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid samples found.\n"
                "Check annotations_dir path and XML structure."
            )

        if verbose:
            pos = sum(1 for s in self.samples if s["label"] == 1.0)
            neg = sum(1 for s in self.samples if s["label"] == 0.0)
            print("[JAADCrossingClipDataset]")
            print(f"  clips_dir       : {clips_dir}")
            print(f"  annotations_dir : {annotations_dir}")
            print(f"  num_frames      : {num_frames}")
            print(f"  frame_stride    : {frame_stride}")
            print(f"  sample_stride   : {sample_stride}")
            print(f"  total samples   : {len(self.samples)}")
            print(f"  positive        : {pos}")
            print(f"  negative        : {neg}")
            print()

    def __len__(self):
        return len(self.samples)

    def _read_clip(self, video_path: str, end_frame: int):
        frame_indices = [
            end_frame - (self.num_frames - 1 - i) * self.frame_stride
            for i in range(self.num_frames)
        ]

        cap = cv2.VideoCapture(video_path)
        frames = []

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"Failed to read frame {fi} from {video_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        return torch.stack(frames, dim=0)  # [T, 3, H, W]

    def __getitem__(self, idx):
        meta = self.samples[idx]
        video = self._read_clip(meta["video_path"], meta["end_frame"])

        return {
            "video": video,
            "label": torch.tensor(meta["label"], dtype=torch.float32),
            "video_path": meta["video_path"],
            "xml_path": meta["xml_path"],
            "end_frame": meta["end_frame"],
        }


# ============================================================
# Losses
# ============================================================
def min_of_k_bce_loss(branch_logits: torch.Tensor, labels: torch.Tensor):
    """
    branch_logits: [B, K, 1]
    labels       : [B, 1]

    Returns:
        loss_scalar
        best_branch_idx: [B]
        per_sample_min_loss: [B]
    """
    labels_exp = labels.unsqueeze(1).expand(-1, branch_logits.size(1), -1)  # [B, K, 1]
    per_branch_loss = F.binary_cross_entropy_with_logits(
        branch_logits,
        labels_exp,
        reduction="none"
    ).squeeze(-1)  # [B, K]

    min_loss, best_branch_idx = per_branch_loss.min(dim=1)  # [B], [B]
    loss = min_loss.mean()

    return loss, best_branch_idx, min_loss


def aggregation_bce_loss(agg_prob: torch.Tensor, labels: torch.Tensor):
    """
    agg_prob: [B, 1]
    labels  : [B, 1]
    """
    agg_prob = agg_prob.clamp(min=1e-6, max=1.0 - 1e-6)
    return F.binary_cross_entropy(agg_prob, labels)


def diversity_loss_cosine(future_feats: torch.Tensor):
    """
    Penalize branch collapse.

    future_feats: [B, K, D]

    Lower is better.
    """
    B, K, D = future_feats.shape
    if K <= 1:
        return future_feats.new_tensor(0.0)

    x = F.normalize(future_feats, dim=-1)            # [B, K, D]
    sim = torch.matmul(x, x.transpose(1, 2))         # [B, K, K]

    eye = torch.eye(K, device=sim.device).unsqueeze(0)  # [1, K, K]
    offdiag = sim * (1.0 - eye)

    num_pairs = K * K - K
    loss = offdiag.sum() / max(B * num_pairs, 1)
    return loss


# ============================================================
# Logging helpers
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
# Train / Valid
# ============================================================
def run_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    log_prefix="Train",
    train_mode=True,
    lambda_mok=1.0,
    lambda_agg=1.0,
    lambda_div=0.05,
    log_interval=50,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss_meter = AverageMeter()
    mok_loss_meter = AverageMeter()
    agg_loss_meter = AverageMeter()
    div_loss_meter = AverageMeter()

    all_probs = []
    all_labels = []
    branch_usage = torch.zeros(model.num_futures, dtype=torch.long)

    start_time = time.time()
    num_batches = len(loader)

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            video = batch["video"].to(device, non_blocking=True)                # [B, T, 3, H, W]
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)  # [B, 1]

            out = model(video)
            branch_logits = out["branch_logits"]   # [B, K, 1]
            agg_prob = out["agg_prob"]             # [B, 1]
            future_feats = out["future_feats"]     # [B, K, D]
            best_branch = out["best_branch"]       # [B, 1] from max prob

            mok_loss, best_branch_mok, _ = min_of_k_bce_loss(branch_logits, labels)
            agg_loss = aggregation_bce_loss(agg_prob, labels)
            div_loss = diversity_loss_cosine(future_feats)

            total_loss = (
                lambda_mok * mok_loss +
                lambda_agg * agg_loss +
                lambda_div * div_loss
            )

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            bs = video.size(0)

            total_loss_meter.update(total_loss.item(), bs)
            mok_loss_meter.update(mok_loss.item(), bs)
            agg_loss_meter.update(agg_loss.item(), bs)
            div_loss_meter.update(div_loss.item(), bs)

            all_probs.append(agg_prob.detach().cpu())
            all_labels.append(labels.detach().cpu())

            for idx in best_branch_mok.detach().cpu().tolist():
                branch_usage[idx] += 1

            elapsed = time.time() - start_time
            avg_batch_time = elapsed / batch_idx
            eta = (num_batches - batch_idx) * avg_batch_time

            if (batch_idx % log_interval == 0) or (batch_idx == num_batches):
                batch_metrics = binary_metrics_from_probs(
                    agg_prob.detach().cpu(),
                    labels.detach().cpu()
                )
                print(
                    f"\r[{log_prefix}] Epoch {epoch:03d}/{total_epochs:03d} | "
                    f"Batch {batch_idx:04d}/{num_batches:04d} | "
                    f"Loss {total_loss_meter.avg:.4f} | "
                    f"MoK {mok_loss_meter.avg:.4f} | "
                    f"Agg {agg_loss_meter.avg:.4f} | "
                    f"Div {div_loss_meter.avg:.4f} | "
                    f"Acc {batch_metrics['acc']:.4f} | "
                    f"Prec {batch_metrics['precision']:.4f} | "
                    f"Rec {batch_metrics['recall']:.4f} | "
                    f"F1 {batch_metrics['f1']:.4f} | "
                    f"ETA {format_seconds(eta)}",
                    end=""
                )

    print()

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = binary_metrics_from_probs(all_probs, all_labels)
    metrics["loss"] = total_loss_meter.avg
    metrics["mok_loss"] = mok_loss_meter.avg
    metrics["agg_loss"] = agg_loss_meter.avg
    metrics["div_loss"] = div_loss_meter.avg
    metrics["branch_usage"] = branch_usage.tolist()

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

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
        "--temporal_ckpt",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\temporal_crossing\best_epoch_005_valF1_0.9777_valAcc_0.9706.pth"
    )

    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--sample_stride", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--freeze_temporal", action="store_true", default=False)

    parser.add_argument("--num_futures", type=int, default=3)
    parser.add_argument("--future_dim", type=int, default=256)

    parser.add_argument("--lr_temporal", type=float, default=5e-5)
    parser.add_argument("--lr_new", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_mok", type=float, default=1.0)
    parser.add_argument("--lambda_agg", type=float, default=1.0)
    parser.add_argument("--lambda_div", type=float, default=0.05)

    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_crossing"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    full_dataset = JAADCrossingClipDataset(
        clips_dir=args.clips_dir,
        annotations_dir=args.annotations_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        sample_stride=args.sample_stride,
        verbose=True,
    )

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
    # Model
    # --------------------------------------------------------
    model = MultiFutureCrossingModel(
        backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
        backbone_dim=768,
        temporal_dim=768,
        future_dim=args.future_dim,
        num_futures=args.num_futures,
        freeze_backbone=args.freeze_backbone,
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
    ).to(device)

    model.load_temporal_checkpoint(
        ckpt_path=args.temporal_ckpt,
        load_backbone=True,
        load_temporal=True,
        verbose=True,
    )

    if args.freeze_temporal:
        for p in model.temporal_encoder.parameters():
            p.requires_grad = False

    print_trainable_modules(model)
    print(f"[Trainable Params] {count_trainable_params(model):,}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    param_groups = []

    temporal_params = [p for p in model.temporal_encoder.parameters() if p.requires_grad]
    new_params = []
    new_params += [p for p in model.future_predictor.parameters() if p.requires_grad]
    new_params += [p for p in model.branch_event_head.parameters() if p.requires_grad]

    if len(temporal_params) > 0:
        param_groups.append({
            "params": temporal_params,
            "lr": args.lr_temporal,
        })

    if len(new_params) > 0:
        param_groups.append({
            "params": new_params,
            "lr": args.lr_new,
        })

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    csv_path = os.path.join(args.save_dir, "metrics.csv")
    best_val_f1 = -1.0
    global_start = time.time()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print("\n" + "=" * 100)
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] START")
        print("=" * 100)

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_prefix="Train",
            train_mode=True,
            lambda_mok=args.lambda_mok,
            lambda_agg=args.lambda_agg,
            lambda_div=args.lambda_div,
            log_interval=args.log_interval,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_prefix="Valid",
            train_mode=False,
            lambda_mok=args.lambda_mok,
            lambda_agg=args.lambda_agg,
            lambda_div=args.lambda_div,
            log_interval=args.log_interval,
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - global_start
        avg_epoch_time = total_elapsed / epoch
        total_eta = (args.epochs - epoch) * avg_epoch_time

        print("\n" + "-" * 100)
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"Time {format_seconds(epoch_time)} | "
            f"Total Elapsed {format_seconds(total_elapsed)} | "
            f"Remaining {format_seconds(total_eta)}"
        )
        print(
            f"[Train] Loss {train_metrics['loss']:.4f} | "
            f"MoK {train_metrics['mok_loss']:.4f} | "
            f"Agg {train_metrics['agg_loss']:.4f} | "
            f"Div {train_metrics['div_loss']:.4f} | "
            f"Acc {train_metrics['acc']:.4f} | "
            f"Prec {train_metrics['precision']:.4f} | "
            f"Rec {train_metrics['recall']:.4f} | "
            f"F1 {train_metrics['f1']:.4f} | "
            f"TP {train_metrics['tp']} TN {train_metrics['tn']} "
            f"FP {train_metrics['fp']} FN {train_metrics['fn']} | "
            f"BranchUsage {train_metrics['branch_usage']}"
        )
        print(
            f"[Valid] Loss {val_metrics['loss']:.4f} | "
            f"MoK {val_metrics['mok_loss']:.4f} | "
            f"Agg {val_metrics['agg_loss']:.4f} | "
            f"Div {val_metrics['div_loss']:.4f} | "
            f"Acc {val_metrics['acc']:.4f} | "
            f"Prec {val_metrics['precision']:.4f} | "
            f"Rec {val_metrics['recall']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"TP {val_metrics['tp']} TN {val_metrics['tn']} "
            f"FP {val_metrics['fp']} FN {val_metrics['fn']} | "
            f"BranchUsage {val_metrics['branch_usage']}"
        )
        print("-" * 100)

        # save per-epoch checkpoint
        epoch_ckpt_path = os.path.join(
            args.save_dir,
            f"epoch_{epoch:03d}_"
            f"valF1_{val_metrics['f1']:.4f}_"
            f"valAcc_{val_metrics['acc']:.4f}.pth"
        )
        save_checkpoint(
            save_path=epoch_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
        )
        print(f"[Checkpoint Saved] {epoch_ckpt_path}")

        latest_ckpt_path = os.path.join(args.save_dir, "latest.pth")
        save_checkpoint(
            save_path=latest_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
        )
        print(f"[Latest Updated] {latest_ckpt_path}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_ckpt_path = os.path.join(
                args.save_dir,
                f"best_epoch_{epoch:03d}_valF1_{val_metrics['f1']:.4f}_valAcc_{val_metrics['acc']:.4f}.pth"
            )
            save_checkpoint(
                save_path=best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
            )
            print(f"[Best Updated] {best_ckpt_path}")

        row = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 4),
            "total_elapsed_sec": round(total_elapsed, 4),

            "train_loss": round(train_metrics["loss"], 6),
            "train_mok_loss": round(train_metrics["mok_loss"], 6),
            "train_agg_loss": round(train_metrics["agg_loss"], 6),
            "train_div_loss": round(train_metrics["div_loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "train_precision": round(train_metrics["precision"], 6),
            "train_recall": round(train_metrics["recall"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "train_tp": train_metrics["tp"],
            "train_tn": train_metrics["tn"],
            "train_fp": train_metrics["fp"],
            "train_fn": train_metrics["fn"],
            "train_branch_usage": str(train_metrics["branch_usage"]),

            "val_loss": round(val_metrics["loss"], 6),
            "val_mok_loss": round(val_metrics["mok_loss"], 6),
            "val_agg_loss": round(val_metrics["agg_loss"], 6),
            "val_div_loss": round(val_metrics["div_loss"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_tp": val_metrics["tp"],
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_branch_usage": str(val_metrics["branch_usage"]),
        }
        append_log_csv(csv_path, row)
        print(f"[Metrics Logged] {csv_path}")

    print("\n" + "=" * 100)
    print(f"[Training Done] Best Val F1 = {best_val_f1:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()