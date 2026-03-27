# Stage 1
# C:\Users\IIPL02\Desktop\Vision Prediction\train_temporal_event.py
# Spatial Backbone + Transformer+Volterra Frame Encoder + 2-Scale Temporal Mamba Encoder
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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from models.temporal_event_model import TemporalEventModel


# ============================================================
# Utility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_modules(model):
    print("\n[Trainable Parameters]")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:80s} {tuple(p.shape)}")
    print()


# ============================================================
# Metrics
# ============================================================
def binary_classification_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    """
    logits: [B, 1]
    labels: [B, 1]
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct = (preds == labels).sum().item()
    total = labels.numel()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / max(total, 1)
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
# Exact JAAD XML parser for crossing labels
# ============================================================
class JAADCrossingParser:
    """
    Parse exact JAAD pedestrian annotation XML from:
        JAAD annotations/annotations/video_xxxx.xml

    Label rule for each frame:
        - if any visible pedestrian has cross == "crossing" -> label 1
        - else if at least one visible pedestrian has cross == "not-crossing" -> label 0
        - else -> unlabeled / skipped

    Positive wins if mixed.
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
    Create sliding video clips:
        frames [t-(T-1)*frame_stride, ..., t]

    Label:
        frame-level crossing label at the last frame t

    Returns:
        {
            "video": [T, 3, H, W],
            "label": scalar float tensor,
            "video_path": str,
            "xml_path": str,
            "end_frame": int
        }
    """

    def __init__(
        self,
        clips_dir: str,
        annotations_dir: str,
        num_frames: int = 8,
        image_size: int = 224,
        frame_stride: int = 1,
        sample_stride: int = 1,
        verbose: bool = True,
    ):
        super().__init__()

        self.clips_dir = clips_dir
        self.annotations_dir = annotations_dir
        self.num_frames = num_frames
        self.image_size = image_size
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
                "Check that annotations_dir points to JAAD annotations/annotations"
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

        video = torch.stack(frames, dim=0)  # [T, 3, H, W]
        return video

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
# Progress printer
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


# ============================================================
# Train / Validate
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()

    loss_meter = AverageMeter()
    all_logits = []
    all_labels = []

    start_time = time.time()
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        video = batch["video"].to(device, non_blocking=True)              # [B, T, 3, H, W]
        label = batch["label"].to(device, non_blocking=True).unsqueeze(1) # [B, 1]

        out = model(video)
        logits = out["logits"]

        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = video.size(0)
        loss_meter.update(loss.item(), bs)

        all_logits.append(logits.detach().cpu())
        all_labels.append(label.detach().cpu())

        elapsed = time.time() - start_time
        avg_batch_time = elapsed / batch_idx
        remain_batches = num_batches - batch_idx
        eta = remain_batches * avg_batch_time

        batch_metrics = binary_classification_metrics_from_logits(
            logits.detach().cpu(),
            label.detach().cpu()
        )

        print(
            f"\r[Train] Epoch {epoch:03d}/{total_epochs:03d} | "
            f"Batch {batch_idx:04d}/{num_batches:04d} | "
            f"Loss {loss_meter.avg:.4f} | "
            f"Acc {batch_metrics['acc']:.4f} | "
            f"Prec {batch_metrics['precision']:.4f} | "
            f"Rec {batch_metrics['recall']:.4f} | "
            f"F1 {batch_metrics['f1']:.4f} | "
            f"ETA {format_seconds(eta)}",
            end=""
        )

    print()

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = binary_classification_metrics_from_logits(all_logits, all_labels)
    metrics["loss"] = loss_meter.avg
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch, total_epochs):
    model.eval()

    loss_meter = AverageMeter()
    all_logits = []
    all_labels = []

    start_time = time.time()
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        video = batch["video"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        out = model(video)
        logits = out["logits"]

        loss = criterion(logits, label)

        bs = video.size(0)
        loss_meter.update(loss.item(), bs)

        all_logits.append(logits.detach().cpu())
        all_labels.append(label.detach().cpu())

        elapsed = time.time() - start_time
        avg_batch_time = elapsed / batch_idx
        remain_batches = num_batches - batch_idx
        eta = remain_batches * avg_batch_time

        batch_metrics = binary_classification_metrics_from_logits(
            logits.detach().cpu(),
            label.detach().cpu()
        )

        print(
            f"\r[Valid] Epoch {epoch:03d}/{total_epochs:03d} | "
            f"Batch {batch_idx:04d}/{num_batches:04d} | "
            f"Loss {loss_meter.avg:.4f} | "
            f"Acc {batch_metrics['acc']:.4f} | "
            f"Prec {batch_metrics['precision']:.4f} | "
            f"Rec {batch_metrics['recall']:.4f} | "
            f"F1 {batch_metrics['f1']:.4f} | "
            f"ETA {format_seconds(eta)}",
            end=""
        )

    print()

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = binary_classification_metrics_from_logits(all_logits, all_labels)
    metrics["loss"] = loss_meter.avg
    return metrics


# ============================================================
# Checkpoint / Logging
# ============================================================
def save_checkpoint(save_path, epoch, model, optimizer, train_metrics, val_metrics, args):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
        "temporal_encoder_type": args.temporal_encoder_type,
        "temporal_pooling": args.temporal_pooling,
        "backbone_name": args.backbone_name,
        "backbone_dim": args.backbone_dim,
        "frame_feature_dim": args.frame_feature_dim,
        "temporal_feature_dim": args.temporal_feature_dim,
    }
    torch.save(ckpt, save_path)


def append_log_csv(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
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

    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--sample_stride", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    # --------------------------------------------------------
    # Train setup
    # --------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--freeze_backbone", type=int, default=1)

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\temporal_crossing_mamba2scale"
    )

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="convnextv2_tiny.fcmae_ft_in22k_in1k"
    )
    parser.add_argument("--backbone_dim", type=int, default=768)

    # --------------------------------------------------------
    # Transformer + Volterra Frame Encoder
    # --------------------------------------------------------
    parser.add_argument("--frame_feature_dim", type=int, default=768)
    parser.add_argument("--frame_encoder_num_heads", type=int, default=8)
    parser.add_argument("--frame_encoder_num_layers", type=int, default=2)
    parser.add_argument("--frame_encoder_ff_dim", type=int, default=1536)
    parser.add_argument("--frame_encoder_dropout", type=float, default=0.1)

    parser.add_argument("--frame_encoder_use_volterra", type=int, default=1)
    parser.add_argument("--frame_encoder_volterra_rank", type=int, default=16)
    parser.add_argument("--frame_encoder_volterra_alpha", type=float, default=1.0)

    # --------------------------------------------------------
    # 2-Scale Temporal Mamba Encoder
    # --------------------------------------------------------
    parser.add_argument("--temporal_encoder_type", type=str, default="mamba_2scale")
    parser.add_argument("--temporal_feature_dim", type=int, default=768)
    parser.add_argument("--temporal_mamba_num_layers", type=int, default=2)
    parser.add_argument("--temporal_mamba_state_dim", type=int, default=16)
    parser.add_argument("--temporal_mamba_conv_kernel", type=int, default=4)
    parser.add_argument("--temporal_mamba_expand", type=int, default=2)
    parser.add_argument("--temporal_mamba_dropout", type=float, default=0.1)
    parser.add_argument("--temporal_mamba_fusion", type=str, default="concat_proj")
    parser.add_argument("--temporal_mamba_local_window", type=int, default=4)
    parser.add_argument("--temporal_pooling", type=str, default="last")

    # --------------------------------------------------------
    # Event head
    # --------------------------------------------------------
    parser.add_argument("--event_hidden_dim", type=int, default=256)
    parser.add_argument("--event_dropout", type=float, default=0.1)

    args = parser.parse_args()

    args.freeze_backbone = bool(args.freeze_backbone)
    args.frame_encoder_use_volterra = bool(args.frame_encoder_use_volterra)

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
    # Compute class balance
    # --------------------------------------------------------
    train_labels = []
    for idx in train_set.indices:
        train_labels.append(full_dataset.samples[idx]["label"])

    num_pos = sum(1 for x in train_labels if x == 1.0)
    num_neg = sum(1 for x in train_labels if x == 0.0)

    print(f"[Train Labels] pos={num_pos}, neg={num_neg}")

    if num_pos > 0:
        pos_weight_value = num_neg / max(num_pos, 1)
    else:
        pos_weight_value = 1.0

    print(f"[Loss] BCEWithLogitsLoss pos_weight={pos_weight_value:.4f}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = TemporalEventModel(
        backbone_name=args.backbone_name,
        backbone_dim=args.backbone_dim,
        freeze_backbone=args.freeze_backbone,

        # [2] Transformer + Volterra Frame Encoder
        frame_feature_dim=args.frame_feature_dim,
        frame_encoder_num_heads=args.frame_encoder_num_heads,
        frame_encoder_num_layers=args.frame_encoder_num_layers,
        frame_encoder_ff_dim=args.frame_encoder_ff_dim,
        frame_encoder_dropout=args.frame_encoder_dropout,
        frame_encoder_use_volterra=args.frame_encoder_use_volterra,
        frame_encoder_volterra_rank=args.frame_encoder_volterra_rank,
        frame_encoder_volterra_alpha=args.frame_encoder_volterra_alpha,

        # [3] 2-Scale Temporal Mamba Encoder
        temporal_encoder_type=args.temporal_encoder_type,
        temporal_mamba_dim=args.temporal_feature_dim,
        temporal_mamba_num_layers=args.temporal_mamba_num_layers,
        temporal_mamba_state_dim=args.temporal_mamba_state_dim,
        temporal_mamba_conv_kernel=args.temporal_mamba_conv_kernel,
        temporal_mamba_expand=args.temporal_mamba_expand,
        temporal_mamba_dropout=args.temporal_mamba_dropout,
        temporal_mamba_fusion=args.temporal_mamba_fusion,
        temporal_mamba_local_window=args.temporal_mamba_local_window,
        temporal_pooling=args.temporal_pooling,

        # Event head
        event_hidden_dim=args.event_hidden_dim,
        event_dropout=args.event_dropout,
    ).to(device)

    print("\n[Model Config]")
    print(f"  backbone_name                  : {args.backbone_name}")
    print(f"  backbone_dim                   : {args.backbone_dim}")
    print(f"  freeze_backbone                : {args.freeze_backbone}")
    print(f"  frame_feature_dim              : {args.frame_feature_dim}")
    print(f"  frame_encoder_num_heads        : {args.frame_encoder_num_heads}")
    print(f"  frame_encoder_num_layers       : {args.frame_encoder_num_layers}")
    print(f"  frame_encoder_ff_dim           : {args.frame_encoder_ff_dim}")
    print(f"  frame_encoder_dropout          : {args.frame_encoder_dropout}")
    print(f"  frame_encoder_use_volterra     : {args.frame_encoder_use_volterra}")
    print(f"  frame_encoder_volterra_rank    : {args.frame_encoder_volterra_rank}")
    print(f"  frame_encoder_volterra_alpha   : {args.frame_encoder_volterra_alpha}")
    print(f"  temporal_encoder_type          : {args.temporal_encoder_type}")
    print(f"  temporal_feature_dim           : {args.temporal_feature_dim}")
    print(f"  temporal_mamba_num_layers      : {args.temporal_mamba_num_layers}")
    print(f"  temporal_mamba_state_dim       : {args.temporal_mamba_state_dim}")
    print(f"  temporal_mamba_conv_kernel     : {args.temporal_mamba_conv_kernel}")
    print(f"  temporal_mamba_expand          : {args.temporal_mamba_expand}")
    print(f"  temporal_mamba_dropout         : {args.temporal_mamba_dropout}")
    print(f"  temporal_mamba_fusion          : {args.temporal_mamba_fusion}")
    print(f"  temporal_mamba_local_window    : {args.temporal_mamba_local_window}")
    print(f"  temporal_pooling               : {args.temporal_pooling}")
    print(f"  event_hidden_dim               : {args.event_hidden_dim}")
    print(f"  event_dropout                  : {args.event_dropout}")
    print()

    print_trainable_modules(model)
    print(f"[Trainable Params] {count_trainable_params(model):,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Logging setup
    # --------------------------------------------------------
    csv_path = os.path.join(args.save_dir, "metrics.csv")
    best_val_f1 = -1.0
    global_start = time.time()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print("\n" + "=" * 90)
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] START")
        print("=" * 90)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - global_start
        avg_epoch_time = total_elapsed / epoch
        remain_epochs = args.epochs - epoch
        total_eta = remain_epochs * avg_epoch_time

        print("\n" + "-" * 90)
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"Time {format_seconds(epoch_time)} | "
            f"Total Elapsed {format_seconds(total_elapsed)} | "
            f"Remaining {format_seconds(total_eta)}"
        )
        print(
            f"[Train] Loss {train_metrics['loss']:.4f} | "
            f"Acc {train_metrics['acc']:.4f} | "
            f"Prec {train_metrics['precision']:.4f} | "
            f"Rec {train_metrics['recall']:.4f} | "
            f"F1 {train_metrics['f1']:.4f} | "
            f"TP {train_metrics['tp']} TN {train_metrics['tn']} "
            f"FP {train_metrics['fp']} FN {train_metrics['fn']}"
        )
        print(
            f"[Valid] Loss {val_metrics['loss']:.4f} | "
            f"Acc {val_metrics['acc']:.4f} | "
            f"Prec {val_metrics['precision']:.4f} | "
            f"Rec {val_metrics['recall']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"TP {val_metrics['tp']} TN {val_metrics['tn']} "
            f"FP {val_metrics['fp']} FN {val_metrics['fn']}"
        )
        print("-" * 90)

        # Save per-epoch checkpoint
        epoch_ckpt_path = os.path.join(
            args.save_dir,
            f"epoch_{epoch:03d}_"
            f"trainF1_{train_metrics['f1']:.4f}_"
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

        # Save latest
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

        # Save best
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

        # Save metrics CSV
        row = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 4),
            "total_elapsed_sec": round(total_elapsed, 4),

            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "train_precision": round(train_metrics["precision"], 6),
            "train_recall": round(train_metrics["recall"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "train_tp": train_metrics["tp"],
            "train_tn": train_metrics["tn"],
            "train_fp": train_metrics["fp"],
            "train_fn": train_metrics["fn"],

            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_tp": val_metrics["tp"],
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],

            "temporal_encoder_type": args.temporal_encoder_type,
            "temporal_pooling": args.temporal_pooling,
            "frame_encoder_use_volterra": int(args.frame_encoder_use_volterra),
            "frame_encoder_volterra_rank": args.frame_encoder_volterra_rank,
            "temporal_feature_dim": args.temporal_feature_dim,
            "temporal_mamba_num_layers": args.temporal_mamba_num_layers,
            "temporal_mamba_state_dim": args.temporal_mamba_state_dim,
            "temporal_mamba_local_window": args.temporal_mamba_local_window,
        }
        append_log_csv(csv_path, row)
        print(f"[Metrics Logged] {csv_path}")

    print("\n" + "=" * 90)
    print(f"[Training Done] Best Val F1 = {best_val_f1:.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()


"""
[1] backbone
[2] transformer + volterra
[3] 2-scale temporal mamba
[4] event head

precision = 0.9806

recall = 0.9538

F1 = 0.9670
"""
    # -< train_multi_future_decision.py에 연결