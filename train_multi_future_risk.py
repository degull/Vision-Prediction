# 5. Risk Head 단계
import os
import csv
import cv2
import time
import math
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

from models.future_model_decision import MultiFutureCrossingDecisionModel


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
# Metrics
# ============================================================
def binary_metrics_from_probs(probs: torch.Tensor, labels: torch.Tensor):
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


def masked_binary_accuracy_from_logits(logits, labels, valid_mask):
    valid_mask = valid_mask.bool().view(-1)
    if valid_mask.sum().item() == 0:
        return 0.0

    probs = torch.sigmoid(logits.view(-1)[valid_mask])
    gt = labels.view(-1)[valid_mask]
    preds = (probs >= 0.5).float()
    return (preds == gt).float().mean().item()


def masked_regression_metrics(preds, targets, valid_mask):
    valid_mask = valid_mask.bool().view(-1)
    if valid_mask.sum().item() == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "acc05": 0.0,
        }

    p = preds.view(-1)[valid_mask]
    t = targets.view(-1)[valid_mask]

    mae = torch.abs(p - t).mean().item()
    rmse = torch.sqrt(((p - t) ** 2).mean()).item()

    bin_pred = (p >= 0.5).float()
    bin_tgt = (t >= 0.5).float()
    acc05 = (bin_pred == bin_tgt).float().mean().item()

    return {
        "mae": mae,
        "rmse": rmse,
        "acc05": acc05,
    }


# ============================================================
# JAAD Parsers
# ============================================================
class JAADMainPedParser:
    """
    Parse main annotations/*.xml

    Returns:
        list of dict:
        [
            {
                "track_id": str,
                "old_id": str,
                "frames": {frame_id: 0.0 or 1.0}
            },
            ...
        ]
    """
    def __init__(self):
        self.pos_label = "crossing"
        self.neg_label = "not-crossing"

    def parse(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        track_list = []

        for track in root.findall(".//track"):
            label_name = track.attrib.get("label", "").strip().lower()
            if label_name not in ["pedestrian", "ped"]:
                continue

            track_id = track.attrib.get("id", "").strip()
            old_id = track.attrib.get("old_id", "").strip()

            frames = {}

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
                if cross_value == self.pos_label:
                    frames[frame_id] = 1.0
                elif cross_value == self.neg_label:
                    frames[frame_id] = 0.0

            if len(frames) > 0:
                track_list.append({
                    "track_id": track_id,
                    "old_id": old_id,
                    "frames": frames,
                })

        return track_list


class JAADAttributesParser:
    """
    Parse annotations_attributes/*.xml

    Returns:
        {
            "decision_points": [int, int, ...]
        }
    """
    def parse(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def _to_int(v, default=-1):
            try:
                return int(v)
            except Exception:
                return default

        decision_points = []

        for ped in root.findall(".//pedestrian"):
            dp = _to_int(ped.attrib.get("decision_point", "-1"), -1)
            if dp >= 0:
                decision_points.append(dp)

        decision_points = sorted(decision_points)

        return {
            "decision_points": decision_points
        }


# ============================================================
# Dataset with pseudo risk label
# ============================================================
class JAADCrossingRiskDataset(Dataset):
    """
    Adds weak/pseudo risk supervision.

    risk_label heuristic:
      - if crossing_label == 1 at end_frame: risk = 1.0
      - else if a decision point exists within early_horizon:
            risk = 1 - delta / early_horizon
        where delta = decision_point - end_frame
      - else:
            risk = 0.0

    This is weak supervision using the currently available JAAD annotations.
    """
    def __init__(
        self,
        clips_dir,
        annotations_dir,
        attributes_dir,
        num_frames=8,
        image_size=224,
        frame_stride=1,
        sample_stride=8,
        early_horizon=30,
        verbose=True,
    ):
        super().__init__()

        self.clips_dir = clips_dir
        self.annotations_dir = annotations_dir
        self.attributes_dir = attributes_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.sample_stride = sample_stride
        self.early_horizon = early_horizon

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.main_parser = JAADMainPedParser()
        self.attr_parser = JAADAttributesParser()

        self.samples = []

        video_files = sorted([f for f in os.listdir(clips_dir) if f.lower().endswith(".mp4")])

        total_parsed_tracks = 0
        total_videos_used = 0
        total_decision_points = 0

        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(clips_dir, video_file)

            main_xml = os.path.join(annotations_dir, f"{video_name}.xml")
            attr_xml = os.path.join(attributes_dir, f"{video_name}_attributes.xml")

            if not (os.path.exists(main_xml) and os.path.exists(attr_xml)):
                continue

            try:
                track_data = self.main_parser.parse(main_xml)
                attr_data = self.attr_parser.parse(attr_xml)
            except Exception as e:
                print(f"[WARN] Parse failed for {video_name}: {e}")
                continue

            decision_points = attr_data["decision_points"]
            total_decision_points += len(decision_points)

            added_samples_this_video = 0
            total_parsed_tracks += len(track_data)

            for info in track_data:
                track_id = info["track_id"]
                old_id = info["old_id"]

                valid_frames = sorted(info["frames"].keys())

                for end_frame in valid_frames[::sample_stride]:
                    start_frame = end_frame - (num_frames - 1) * frame_stride
                    if start_frame < 0:
                        continue

                    crossing_label = float(info["frames"][end_frame])

                    early_valid = 1.0 if len(decision_points) > 0 else 0.0

                    early_label = 0.0
                    nearest_delta = None

                    if len(decision_points) > 0:
                        future_deltas = []
                        for dp in decision_points:
                            delta = dp - end_frame
                            if delta >= 0:
                                future_deltas.append(delta)
                            if 0 <= delta <= early_horizon:
                                early_label = 1.0

                        if len(future_deltas) > 0:
                            nearest_delta = min(future_deltas)

                    # -----------------------------
                    # Pseudo risk target
                    # -----------------------------
                    if crossing_label >= 1.0:
                        risk_label = 1.0
                        risk_valid = 1.0
                    elif nearest_delta is not None and nearest_delta <= early_horizon:
                        # closer decision point => higher risk
                        risk_label = 1.0 - (nearest_delta / float(max(early_horizon, 1)))
                        risk_label = float(max(0.0, min(1.0, risk_label)))
                        risk_valid = 1.0
                    else:
                        risk_label = 0.0
                        risk_valid = 1.0

                    self.samples.append({
                        "video_path": video_path,
                        "ped_id": old_id if old_id else track_id,
                        "old_id": old_id,
                        "track_id": track_id,
                        "end_frame": end_frame,
                        "crossing_label": crossing_label,
                        "early_label": early_label,
                        "early_valid": early_valid,
                        "risk_label": risk_label,
                        "risk_valid": risk_valid,
                        "decision_points": decision_points,
                    })
                    added_samples_this_video += 1

            if added_samples_this_video > 0:
                total_videos_used += 1

            if verbose:
                print(
                    f"[DEBUG] {video_name} | "
                    f"parsed_tracks={len(track_data)} | "
                    f"decision_points={len(decision_points)} | "
                    f"added_samples={added_samples_this_video}"
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid samples found.\n"
                "Possible causes:\n"
                "1) crossing labels were not found in main annotations\n"
                "2) decision point xml path is wrong\n"
                "3) clips/annotation paths are wrong\n"
            )

        if verbose:
            pos = sum(1 for s in self.samples if s["crossing_label"] == 1.0)
            neg = sum(1 for s in self.samples if s["crossing_label"] == 0.0)
            ev = sum(1 for s in self.samples if s["early_valid"] == 1.0)
            ep = sum(1 for s in self.samples if s["early_label"] == 1.0)

            risk_mean = sum(s["risk_label"] for s in self.samples) / max(len(self.samples), 1)
            risk_pos = sum(1 for s in self.samples if s["risk_label"] >= 0.5)

            print("[JAADCrossingRiskDataset]")
            print(f"  clips_dir            : {clips_dir}")
            print(f"  annotations_dir      : {annotations_dir}")
            print(f"  attributes_dir       : {attributes_dir}")
            print(f"  num_frames           : {num_frames}")
            print(f"  frame_stride         : {frame_stride}")
            print(f"  sample_stride        : {sample_stride}")
            print(f"  early_horizon        : {early_horizon}")
            print(f"  total videos used    : {total_videos_used}")
            print(f"  total parsed tracks  : {total_parsed_tracks}")
            print(f"  total decision pts   : {total_decision_points}")
            print(f"  total samples        : {len(self.samples)}")
            print(f"  crossing positive    : {pos}")
            print(f"  crossing negative    : {neg}")
            print(f"  early valid samples  : {ev}")
            print(f"  early positive       : {ep}")
            print(f"  risk mean            : {risk_mean:.4f}")
            print(f"  risk >= 0.5 samples  : {risk_pos}")
            print()

    def __len__(self):
        return len(self.samples)

    def _read_clip(self, video_path, end_frame):
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
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        video = self._read_clip(meta["video_path"], meta["end_frame"])

        return {
            "video": video,
            "crossing_label": torch.tensor(meta["crossing_label"], dtype=torch.float32),
            "early_label": torch.tensor(meta["early_label"], dtype=torch.float32),
            "early_valid": torch.tensor(meta["early_valid"], dtype=torch.float32),
            "risk_label": torch.tensor(meta["risk_label"], dtype=torch.float32),
            "risk_valid": torch.tensor(meta["risk_valid"], dtype=torch.float32),
            "ped_id": meta["ped_id"],
            "old_id": meta["old_id"],
            "track_id": meta["track_id"],
            "video_path": meta["video_path"],
            "end_frame": meta["end_frame"],
        }


# ============================================================
# Model wrapper: base multi-future model + branch risk head
# ============================================================
class BranchRiskHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, future_feats):
        # future_feats: [B, K, D]
        B, K, D = future_feats.shape
        x = future_feats.reshape(B * K, D)
        logits = self.head(x).reshape(B, K, 1)
        return logits


class MultiFutureRiskModel(nn.Module):
    def __init__(
        self,
        base_model: MultiFutureCrossingDecisionModel,
        future_dim=256,
        risk_hidden_dim=128,
        risk_dropout=0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_futures = base_model.num_futures
        self.branch_risk_head = BranchRiskHead(
            in_dim=future_dim,
            hidden_dim=risk_hidden_dim,
            dropout=risk_dropout,
        )

    def forward(self, video):
        out = self.base_model(video)
        future_feats = out["future_feats"]                    # [B, K, D]
        branch_risk_logits = self.branch_risk_head(future_feats)  # [B, K, 1]
        branch_risk_probs = torch.sigmoid(branch_risk_logits)
        agg_risk_prob = branch_risk_probs.mean(dim=1)         # [B, 1]

        out["branch_risk_logits"] = branch_risk_logits
        out["branch_risk_probs"] = branch_risk_probs
        out["agg_risk_prob"] = agg_risk_prob
        return out


# ============================================================
# Losses
# ============================================================
def masked_bce_with_logits(logits, labels, valid_mask):
    valid_mask = valid_mask.float()
    per_elem = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    masked = per_elem * valid_mask
    denom = valid_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def masked_mse_loss(preds, targets, valid_mask):
    valid_mask = valid_mask.float()
    per_elem = (preds - targets) ** 2
    masked = per_elem * valid_mask
    denom = valid_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def risk_branch_bce_loss(branch_risk_logits, risk_labels):
    # branch_risk_logits: [B, K, 1]
    # risk_labels: [B, 1]
    labels_exp = risk_labels.unsqueeze(1).expand(-1, branch_risk_logits.size(1), -1)
    return F.binary_cross_entropy_with_logits(branch_risk_logits, labels_exp)


def risk_agg_mse_loss(agg_risk_prob, risk_labels, risk_valid):
    return masked_mse_loss(agg_risk_prob, risk_labels, risk_valid)


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
    lambda_risk_branch=1.0,
    lambda_risk_agg=1.0,
    lambda_event_consistency=0.2,
    log_interval=50,
):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss_meter = AverageMeter()
    risk_branch_loss_meter = AverageMeter()
    risk_agg_loss_meter = AverageMeter()
    event_consistency_meter = AverageMeter()

    all_cross_probs = []
    all_cross_labels = []
    all_early_logits = []
    all_early_labels = []
    all_early_valid = []

    all_risk_probs = []
    all_risk_labels = []
    all_risk_valid = []

    start_time = time.time()
    num_batches = len(loader)

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            video = batch["video"].to(device, non_blocking=True)

            crossing_labels = batch["crossing_label"].to(device, non_blocking=True).unsqueeze(1)
            early_labels = batch["early_label"].to(device, non_blocking=True).unsqueeze(1)
            early_valid = batch["early_valid"].to(device, non_blocking=True).unsqueeze(1)

            risk_labels = batch["risk_label"].to(device, non_blocking=True).unsqueeze(1)
            risk_valid = batch["risk_valid"].to(device, non_blocking=True).unsqueeze(1)

            out = model(video)

            agg_prob = out["agg_prob"]                        # crossing prob
            early_logit = out["early_logit"]
            branch_risk_logits = out["branch_risk_logits"]    # [B, K, 1]
            agg_risk_prob = out["agg_risk_prob"]              # [B, 1]

            risk_branch_loss = risk_branch_bce_loss(branch_risk_logits, risk_labels)
            risk_agg_loss = risk_agg_mse_loss(agg_risk_prob, risk_labels, risk_valid)

            # keep branch risk loosely aligned with event probability
            event_consistency = F.l1_loss(agg_risk_prob, agg_prob.detach())

            total_loss = (
                lambda_risk_branch * risk_branch_loss +
                lambda_risk_agg * risk_agg_loss +
                lambda_event_consistency * event_consistency
            )

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = video.size(0)

            total_loss_meter.update(total_loss.item(), bs)
            risk_branch_loss_meter.update(risk_branch_loss.item(), bs)
            risk_agg_loss_meter.update(risk_agg_loss.item(), bs)
            event_consistency_meter.update(event_consistency.item(), bs)

            all_cross_probs.append(agg_prob.detach().cpu())
            all_cross_labels.append(crossing_labels.detach().cpu())

            all_early_logits.append(early_logit.detach().cpu())
            all_early_labels.append(early_labels.detach().cpu())
            all_early_valid.append(early_valid.detach().cpu())

            all_risk_probs.append(agg_risk_prob.detach().cpu())
            all_risk_labels.append(risk_labels.detach().cpu())
            all_risk_valid.append(risk_valid.detach().cpu())

            elapsed = time.time() - start_time
            avg_batch_time = elapsed / batch_idx
            eta = (num_batches - batch_idx) * avg_batch_time

            if (batch_idx % log_interval == 0) or (batch_idx == num_batches):
                batch_cross = binary_metrics_from_probs(
                    agg_prob.detach().cpu(),
                    crossing_labels.detach().cpu()
                )
                batch_early_acc = masked_binary_accuracy_from_logits(
                    early_logit.detach().cpu(),
                    early_labels.detach().cpu(),
                    early_valid.detach().cpu()
                )
                batch_risk = masked_regression_metrics(
                    agg_risk_prob.detach().cpu(),
                    risk_labels.detach().cpu(),
                    risk_valid.detach().cpu()
                )

                print(
                    f"\r[{log_prefix}] Epoch {epoch:03d}/{total_epochs:03d} | "
                    f"Batch {batch_idx:04d}/{num_batches:04d} | "
                    f"Loss {total_loss_meter.avg:.4f} | "
                    f"RiskBCE {risk_branch_loss_meter.avg:.4f} | "
                    f"RiskMSE {risk_agg_loss_meter.avg:.4f} | "
                    f"Cons {event_consistency_meter.avg:.4f} | "
                    f"CrossF1 {batch_cross['f1']:.4f} | "
                    f"EarlyAcc {batch_early_acc:.4f} | "
                    f"RiskMAE {batch_risk['mae']:.4f} | "
                    f"ETA {format_seconds(eta)}",
                    end=""
                )

    print()

    all_cross_probs = torch.cat(all_cross_probs, dim=0)
    all_cross_labels = torch.cat(all_cross_labels, dim=0)

    all_early_logits = torch.cat(all_early_logits, dim=0)
    all_early_labels = torch.cat(all_early_labels, dim=0)
    all_early_valid = torch.cat(all_early_valid, dim=0)

    all_risk_probs = torch.cat(all_risk_probs, dim=0)
    all_risk_labels = torch.cat(all_risk_labels, dim=0)
    all_risk_valid = torch.cat(all_risk_valid, dim=0)

    crossing_metrics = binary_metrics_from_probs(all_cross_probs, all_cross_labels)
    early_acc = masked_binary_accuracy_from_logits(all_early_logits, all_early_labels, all_early_valid)
    risk_metrics = masked_regression_metrics(all_risk_probs, all_risk_labels, all_risk_valid)

    metrics = dict(crossing_metrics)
    metrics["loss"] = total_loss_meter.avg
    metrics["risk_branch_loss"] = risk_branch_loss_meter.avg
    metrics["risk_agg_loss"] = risk_agg_loss_meter.avg
    metrics["event_consistency"] = event_consistency_meter.avg
    metrics["early_acc"] = early_acc
    metrics["risk_mae"] = risk_metrics["mae"]
    metrics["risk_rmse"] = risk_metrics["rmse"]
    metrics["risk_acc05"] = risk_metrics["acc05"]

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_ckpt",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_decision_branchfix\best_branchfix_epoch_003_valF1_0.8818_valEarly_0.8838.pth"
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
    parser.add_argument("--sample_stride", type=int, default=8)
    parser.add_argument("--early_horizon", type=int, default=30)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--freeze_temporal", action="store_true", default=True)
    parser.add_argument("--freeze_future_predictor", action="store_true", default=True)
    parser.add_argument("--freeze_event_heads", action="store_true", default=True)

    parser.add_argument("--num_futures", type=int, default=3)
    parser.add_argument("--future_dim", type=int, default=256)

    parser.add_argument("--risk_hidden_dim", type=int, default=128)
    parser.add_argument("--risk_dropout", type=float, default=0.1)

    parser.add_argument("--lr_risk", type=float, default=1e-4)
    parser.add_argument("--lr_unfrozen", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_risk_branch", type=float, default=1.0)
    parser.add_argument("--lambda_risk_agg", type=float, default=1.0)
    parser.add_argument("--lambda_event_consistency", type=float, default=0.2)

    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_risk"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    full_dataset = JAADCrossingRiskDataset(
        clips_dir=args.clips_dir,
        annotations_dir=args.annotations_dir,
        attributes_dir=args.attributes_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        sample_stride=args.sample_stride,
        early_horizon=args.early_horizon,
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
    # Build base model and load existing best branch-fix ckpt
    # --------------------------------------------------------
    base_model = MultiFutureCrossingDecisionModel(
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
        early_hidden_dim=256,
        early_dropout=0.1,
    ).to(device)

    print(f"[Load Base Checkpoint] {args.base_ckpt}")
    ckpt = torch.load(args.base_ckpt, map_location=device)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Freeze policy
    if args.freeze_temporal:
        for p in base_model.temporal_encoder.parameters():
            p.requires_grad = False

    if args.freeze_future_predictor:
        for p in base_model.future_predictor.parameters():
            p.requires_grad = False

    if args.freeze_event_heads:
        for p in base_model.branch_event_head.parameters():
            p.requires_grad = False
        for p in base_model.early_head.parameters():
            p.requires_grad = False

    model = MultiFutureRiskModel(
        base_model=base_model,
        future_dim=args.future_dim,
        risk_hidden_dim=args.risk_hidden_dim,
        risk_dropout=args.risk_dropout,
    ).to(device)

    # risk head is always trainable
    for p in model.branch_risk_head.parameters():
        p.requires_grad = True

    print_trainable_modules(model)
    print(f"[Trainable Params] {count_trainable_params(model):,}")

    risk_params = [p for p in model.branch_risk_head.parameters() if p.requires_grad]
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("branch_risk_head."):
            continue
        other_params.append(p)

    param_groups = []
    if len(risk_params) > 0:
        param_groups.append({"params": risk_params, "lr": args.lr_risk})
    if len(other_params) > 0:
        param_groups.append({"params": other_params, "lr": args.lr_unfrozen})

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    csv_path = os.path.join(args.save_dir, "metrics.csv")
    best_val_risk_mae = 1e9
    global_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print("\n" + "=" * 110)
        print(f"[Risk Epoch {epoch:03d}/{args.epochs:03d}] START")
        print("=" * 110)

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_prefix="Train",
            train_mode=True,
            lambda_risk_branch=args.lambda_risk_branch,
            lambda_risk_agg=args.lambda_risk_agg,
            lambda_event_consistency=args.lambda_event_consistency,
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
            lambda_risk_branch=args.lambda_risk_branch,
            lambda_risk_agg=args.lambda_risk_agg,
            lambda_event_consistency=args.lambda_event_consistency,
            log_interval=args.log_interval,
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - global_start
        avg_epoch_time = total_elapsed / epoch
        total_eta = (args.epochs - epoch) * avg_epoch_time

        print("\n" + "-" * 110)
        print(
            f"[Risk Epoch {epoch:03d}/{args.epochs:03d}] "
            f"Time {format_seconds(epoch_time)} | "
            f"Total Elapsed {format_seconds(total_elapsed)} | "
            f"Remaining {format_seconds(total_eta)}"
        )
        print(
            f"[Train] Loss {train_metrics['loss']:.4f} | "
            f"RiskBCE {train_metrics['risk_branch_loss']:.4f} | "
            f"RiskMSE {train_metrics['risk_agg_loss']:.4f} | "
            f"Cons {train_metrics['event_consistency']:.4f} | "
            f"CrossAcc {train_metrics['acc']:.4f} | "
            f"CrossF1 {train_metrics['f1']:.4f} | "
            f"EarlyAcc {train_metrics['early_acc']:.4f} | "
            f"RiskMAE {train_metrics['risk_mae']:.4f} | "
            f"RiskRMSE {train_metrics['risk_rmse']:.4f} | "
            f"RiskAcc@0.5 {train_metrics['risk_acc05']:.4f}"
        )
        print(
            f"[Valid] Loss {val_metrics['loss']:.4f} | "
            f"RiskBCE {val_metrics['risk_branch_loss']:.4f} | "
            f"RiskMSE {val_metrics['risk_agg_loss']:.4f} | "
            f"Cons {val_metrics['event_consistency']:.4f} | "
            f"CrossAcc {val_metrics['acc']:.4f} | "
            f"CrossF1 {val_metrics['f1']:.4f} | "
            f"EarlyAcc {val_metrics['early_acc']:.4f} | "
            f"RiskMAE {val_metrics['risk_mae']:.4f} | "
            f"RiskRMSE {val_metrics['risk_rmse']:.4f} | "
            f"RiskAcc@0.5 {val_metrics['risk_acc05']:.4f}"
        )
        print("-" * 110)

        epoch_ckpt_path = os.path.join(
            args.save_dir,
            f"risk_epoch_{epoch:03d}_valRiskMAE_{val_metrics['risk_mae']:.4f}_valF1_{val_metrics['f1']:.4f}.pth"
        )
        save_checkpoint(epoch_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
        print(f"[Checkpoint Saved] {epoch_ckpt_path}")

        latest_ckpt_path = os.path.join(args.save_dir, "latest.pth")
        save_checkpoint(latest_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
        print(f"[Latest Updated] {latest_ckpt_path}")

        if val_metrics["risk_mae"] < best_val_risk_mae:
            best_val_risk_mae = val_metrics["risk_mae"]
            best_ckpt_path = os.path.join(
                args.save_dir,
                f"best_risk_epoch_{epoch:03d}_valRiskMAE_{val_metrics['risk_mae']:.4f}_valF1_{val_metrics['f1']:.4f}.pth"
            )
            save_checkpoint(best_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
            print(f"[Best Updated] {best_ckpt_path}")

        row = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 4),
            "total_elapsed_sec": round(total_elapsed, 4),

            "train_loss": round(train_metrics["loss"], 6),
            "train_risk_branch_loss": round(train_metrics["risk_branch_loss"], 6),
            "train_risk_agg_loss": round(train_metrics["risk_agg_loss"], 6),
            "train_event_consistency": round(train_metrics["event_consistency"], 6),
            "train_cross_acc": round(train_metrics["acc"], 6),
            "train_cross_precision": round(train_metrics["precision"], 6),
            "train_cross_recall": round(train_metrics["recall"], 6),
            "train_cross_f1": round(train_metrics["f1"], 6),
            "train_early_acc": round(train_metrics["early_acc"], 6),
            "train_risk_mae": round(train_metrics["risk_mae"], 6),
            "train_risk_rmse": round(train_metrics["risk_rmse"], 6),
            "train_risk_acc05": round(train_metrics["risk_acc05"], 6),

            "val_loss": round(val_metrics["loss"], 6),
            "val_risk_branch_loss": round(val_metrics["risk_branch_loss"], 6),
            "val_risk_agg_loss": round(val_metrics["risk_agg_loss"], 6),
            "val_event_consistency": round(val_metrics["event_consistency"], 6),
            "val_cross_acc": round(val_metrics["acc"], 6),
            "val_cross_precision": round(val_metrics["precision"], 6),
            "val_cross_recall": round(val_metrics["recall"], 6),
            "val_cross_f1": round(val_metrics["f1"], 6),
            "val_early_acc": round(val_metrics["early_acc"], 6),
            "val_risk_mae": round(val_metrics["risk_mae"], 6),
            "val_risk_rmse": round(val_metrics["risk_rmse"], 6),
            "val_risk_acc05": round(val_metrics["risk_acc05"], 6),
        }
        append_log_csv(csv_path, row)
        print(f"[Metrics Logged] {csv_path}")

    print("\n" + "=" * 110)
    print(f"[Risk Training Done] Best Val Risk MAE = {best_val_risk_mae:.4f}")
    print("=" * 110)


if __name__ == "__main__":
    main()


    