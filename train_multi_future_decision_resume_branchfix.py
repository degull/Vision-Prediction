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
            print(f"{name:70s} {tuple(p.shape)}")
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


# ============================================================
# JAAD Parsers
# ============================================================
class JAADMainPedParser:
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
# Dataset
# ============================================================
class JAADCrossingDecisionDataset(Dataset):
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

                    crossing_label = info["frames"][end_frame]
                    early_valid = 1.0 if len(decision_points) > 0 else 0.0

                    early_label = 0.0
                    if len(decision_points) > 0:
                        for dp in decision_points:
                            delta = dp - end_frame
                            if 0 <= delta <= early_horizon:
                                early_label = 1.0
                                break

                    self.samples.append({
                        "video_path": video_path,
                        "ped_id": old_id if old_id else track_id,
                        "old_id": old_id,
                        "track_id": track_id,
                        "end_frame": end_frame,
                        "crossing_label": crossing_label,
                        "early_label": early_label,
                        "early_valid": early_valid,
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
            raise RuntimeError("No valid samples found.")

        if verbose:
            pos = sum(1 for s in self.samples if s["crossing_label"] == 1.0)
            neg = sum(1 for s in self.samples if s["crossing_label"] == 0.0)
            ev = sum(1 for s in self.samples if s["early_valid"] == 1.0)
            ep = sum(1 for s in self.samples if s["early_label"] == 1.0)

            print("[JAADCrossingDecisionDataset]")
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
            "ped_id": meta["ped_id"],
            "old_id": meta["old_id"],
            "track_id": meta["track_id"],
            "video_path": meta["video_path"],
            "end_frame": meta["end_frame"],
        }


# ============================================================
# Losses
# ============================================================
def compute_per_branch_bce(branch_logits, labels):
    labels_exp = labels.unsqueeze(1).expand(-1, branch_logits.size(1), -1)
    per_branch_loss = F.binary_cross_entropy_with_logits(
        branch_logits,
        labels_exp,
        reduction="none"
    ).squeeze(-1)  # [B, K]
    return per_branch_loss


def min_of_k_bce_loss(branch_logits, labels):
    per_branch_loss = compute_per_branch_bce(branch_logits, labels)
    min_loss, best_branch_idx = per_branch_loss.min(dim=1)
    loss = min_loss.mean()
    return loss, best_branch_idx, min_loss, per_branch_loss


def aggregation_bce_loss(agg_prob, labels):
    agg_prob = agg_prob.clamp(min=1e-6, max=1.0 - 1e-6)
    return F.binary_cross_entropy(agg_prob, labels)


def diversity_loss_cosine(future_feats):
    B, K, D = future_feats.shape
    if K <= 1:
        return future_feats.new_tensor(0.0)

    x = F.normalize(future_feats, dim=-1)
    sim = torch.matmul(x, x.transpose(1, 2))

    eye = torch.eye(K, device=sim.device).unsqueeze(0)
    offdiag = sim * (1.0 - eye)

    num_pairs = K * K - K
    loss = offdiag.sum() / max(B * num_pairs, 1)
    return loss


def masked_bce_with_logits(logits, labels, valid_mask):
    valid_mask = valid_mask.float()
    per_elem = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    masked = per_elem * valid_mask
    denom = valid_mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def soft_branch_balance_loss(per_branch_loss, temperature=0.7):
    """
    per_branch_loss: [B, K]
    branch loss가 낮을수록 그 branch가 선택될 가능성이 높다고 보고
    soft assignment를 만들어 batch 평균 usage가 균등하도록 유도한다.
    """
    assign_prob = torch.softmax(-per_branch_loss / temperature, dim=1)  # [B, K]
    mean_usage = assign_prob.mean(dim=0)  # [K]
    target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    loss = ((mean_usage - target) ** 2).mean()
    return loss, mean_usage, assign_prob


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
    lambda_div=0.15,
    lambda_early=0.5,
    lambda_balance=0.5,
    balance_temp=0.7,
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
    early_loss_meter = AverageMeter()
    balance_loss_meter = AverageMeter()

    all_probs = []
    all_labels = []
    all_early_logits = []
    all_early_labels = []
    all_early_valid = []

    branch_usage = torch.zeros(model.num_futures, dtype=torch.long)

    start_time = time.time()
    num_batches = len(loader)

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            crossing_labels = batch["crossing_label"].to(device, non_blocking=True).unsqueeze(1)
            early_labels = batch["early_label"].to(device, non_blocking=True).unsqueeze(1)
            early_valid = batch["early_valid"].to(device, non_blocking=True).unsqueeze(1)

            out = model(video)

            branch_logits = out["branch_logits"]     # [B, K, 1]
            agg_prob = out["agg_prob"]               # [B, 1]
            future_feats = out["future_feats"]       # [B, K, D]
            early_logit = out["early_logit"]         # [B, 1]

            mok_loss, best_branch_mok, _, per_branch_loss = min_of_k_bce_loss(branch_logits, crossing_labels)
            agg_loss = aggregation_bce_loss(agg_prob, crossing_labels)
            div_loss = diversity_loss_cosine(future_feats)
            early_loss = masked_bce_with_logits(early_logit, early_labels, early_valid)

            balance_loss, mean_usage_soft, _ = soft_branch_balance_loss(
                per_branch_loss,
                temperature=balance_temp,
            )

            total_loss = (
                lambda_mok * mok_loss +
                lambda_agg * agg_loss +
                lambda_div * div_loss +
                lambda_early * early_loss +
                lambda_balance * balance_loss
            )

            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = video.size(0)

            total_loss_meter.update(total_loss.item(), bs)
            mok_loss_meter.update(mok_loss.item(), bs)
            agg_loss_meter.update(agg_loss.item(), bs)
            div_loss_meter.update(div_loss.item(), bs)
            early_loss_meter.update(early_loss.item(), bs)
            balance_loss_meter.update(balance_loss.item(), bs)

            all_probs.append(agg_prob.detach().cpu())
            all_labels.append(crossing_labels.detach().cpu())
            all_early_logits.append(early_logit.detach().cpu())
            all_early_labels.append(early_labels.detach().cpu())
            all_early_valid.append(early_valid.detach().cpu())

            for idx in best_branch_mok.detach().cpu().tolist():
                branch_usage[idx] += 1

            elapsed = time.time() - start_time
            avg_batch_time = elapsed / batch_idx
            eta = (num_batches - batch_idx) * avg_batch_time

            if (batch_idx % log_interval == 0) or (batch_idx == num_batches):
                batch_metrics = binary_metrics_from_probs(
                    agg_prob.detach().cpu(),
                    crossing_labels.detach().cpu()
                )
                batch_early_acc = masked_binary_accuracy_from_logits(
                    early_logit.detach().cpu(),
                    early_labels.detach().cpu(),
                    early_valid.detach().cpu()
                )

                mean_usage_soft_cpu = mean_usage_soft.detach().cpu().tolist()
                mean_usage_soft_str = "[" + ", ".join([f"{v:.3f}" for v in mean_usage_soft_cpu]) + "]"

                print(
                    f"\r[{log_prefix}] Epoch {epoch:03d}/{total_epochs:03d} | "
                    f"Batch {batch_idx:04d}/{num_batches:04d} | "
                    f"Loss {total_loss_meter.avg:.4f} | "
                    f"MoK {mok_loss_meter.avg:.4f} | "
                    f"Agg {agg_loss_meter.avg:.4f} | "
                    f"Div {div_loss_meter.avg:.4f} | "
                    f"Early {early_loss_meter.avg:.4f} | "
                    f"Bal {balance_loss_meter.avg:.4f} | "
                    f"CrossF1 {batch_metrics['f1']:.4f} | "
                    f"EarlyAcc {batch_early_acc:.4f} | "
                    f"SoftUsage {mean_usage_soft_str} | "
                    f"ETA {format_seconds(eta)}",
                    end=""
                )

    print()

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_early_logits = torch.cat(all_early_logits, dim=0)
    all_early_labels = torch.cat(all_early_labels, dim=0)
    all_early_valid = torch.cat(all_early_valid, dim=0)

    crossing_metrics = binary_metrics_from_probs(all_probs, all_labels)
    early_acc = masked_binary_accuracy_from_logits(all_early_logits, all_early_labels, all_early_valid)

    metrics = dict(crossing_metrics)
    metrics["loss"] = total_loss_meter.avg
    metrics["mok_loss"] = mok_loss_meter.avg
    metrics["agg_loss"] = agg_loss_meter.avg
    metrics["div_loss"] = div_loss_meter.avg
    metrics["early_loss"] = early_loss_meter.avg
    metrics["balance_loss"] = balance_loss_meter.avg
    metrics["early_acc"] = early_acc
    metrics["branch_usage"] = branch_usage.tolist()

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_decision\epoch_007_valF1_0.8778_valEarly_0.8805.pth"
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

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--freeze_temporal", action="store_true", default=True)

    parser.add_argument("--num_futures", type=int, default=3)
    parser.add_argument("--future_dim", type=int, default=256)

    parser.add_argument("--lr_temporal", type=float, default=1e-5)
    parser.add_argument("--lr_new", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_mok", type=float, default=1.0)
    parser.add_argument("--lambda_agg", type=float, default=1.0)
    parser.add_argument("--lambda_div", type=float, default=0.15)
    parser.add_argument("--lambda_early", type=float, default=0.5)
    parser.add_argument("--lambda_balance", type=float, default=0.5)
    parser.add_argument("--balance_temp", type=float, default=0.7)

    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_decision_branchfix"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    full_dataset = JAADCrossingDecisionDataset(
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

    model = MultiFutureCrossingDecisionModel(
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

    print(f"[Resume Load] {args.resume_ckpt}")
    ckpt = torch.load(args.resume_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if args.freeze_temporal:
        for p in model.temporal_encoder.parameters():
            p.requires_grad = False

    print_trainable_modules(model)
    print(f"[Trainable Params] {count_trainable_params(model):,}")

    temporal_params = [p for p in model.temporal_encoder.parameters() if p.requires_grad]
    new_params = []
    new_params += [p for p in model.future_predictor.parameters() if p.requires_grad]
    new_params += [p for p in model.branch_event_head.parameters() if p.requires_grad]
    new_params += [p for p in model.early_head.parameters() if p.requires_grad]

    param_groups = []
    if len(temporal_params) > 0:
        param_groups.append({"params": temporal_params, "lr": args.lr_temporal})
    if len(new_params) > 0:
        param_groups.append({"params": new_params, "lr": args.lr_new})

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    csv_path = os.path.join(args.save_dir, "metrics.csv")
    best_val_f1 = -1.0
    global_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print("\n" + "=" * 110)
        print(f"[Branch-Fix Epoch {epoch:03d}/{args.epochs:03d}] START")
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
            lambda_mok=args.lambda_mok,
            lambda_agg=args.lambda_agg,
            lambda_div=args.lambda_div,
            lambda_early=args.lambda_early,
            lambda_balance=args.lambda_balance,
            balance_temp=args.balance_temp,
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
            lambda_early=args.lambda_early,
            lambda_balance=args.lambda_balance,
            balance_temp=args.balance_temp,
            log_interval=args.log_interval,
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - global_start
        avg_epoch_time = total_elapsed / epoch
        total_eta = (args.epochs - epoch) * avg_epoch_time

        print("\n" + "-" * 110)
        print(
            f"[Branch-Fix Epoch {epoch:03d}/{args.epochs:03d}] "
            f"Time {format_seconds(epoch_time)} | "
            f"Total Elapsed {format_seconds(total_elapsed)} | "
            f"Remaining {format_seconds(total_eta)}"
        )
        print(
            f"[Train] Loss {train_metrics['loss']:.4f} | "
            f"MoK {train_metrics['mok_loss']:.4f} | "
            f"Agg {train_metrics['agg_loss']:.4f} | "
            f"Div {train_metrics['div_loss']:.4f} | "
            f"Early {train_metrics['early_loss']:.4f} | "
            f"Bal {train_metrics['balance_loss']:.4f} | "
            f"CrossAcc {train_metrics['acc']:.4f} | "
            f"CrossF1 {train_metrics['f1']:.4f} | "
            f"EarlyAcc {train_metrics['early_acc']:.4f} | "
            f"BranchUsage {train_metrics['branch_usage']}"
        )
        print(
            f"[Valid] Loss {val_metrics['loss']:.4f} | "
            f"MoK {val_metrics['mok_loss']:.4f} | "
            f"Agg {val_metrics['agg_loss']:.4f} | "
            f"Div {val_metrics['div_loss']:.4f} | "
            f"Early {val_metrics['early_loss']:.4f} | "
            f"Bal {val_metrics['balance_loss']:.4f} | "
            f"CrossAcc {val_metrics['acc']:.4f} | "
            f"CrossF1 {val_metrics['f1']:.4f} | "
            f"EarlyAcc {val_metrics['early_acc']:.4f} | "
            f"BranchUsage {val_metrics['branch_usage']}"
        )
        print("-" * 110)

        epoch_ckpt_path = os.path.join(
            args.save_dir,
            f"branchfix_epoch_{epoch:03d}_valF1_{val_metrics['f1']:.4f}_valEarly_{val_metrics['early_acc']:.4f}.pth"
        )
        save_checkpoint(epoch_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
        print(f"[Checkpoint Saved] {epoch_ckpt_path}")

        latest_ckpt_path = os.path.join(args.save_dir, "latest.pth")
        save_checkpoint(latest_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
        print(f"[Latest Updated] {latest_ckpt_path}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_ckpt_path = os.path.join(
                args.save_dir,
                f"best_branchfix_epoch_{epoch:03d}_valF1_{val_metrics['f1']:.4f}_valEarly_{val_metrics['early_acc']:.4f}.pth"
            )
            save_checkpoint(best_ckpt_path, epoch, model, optimizer, train_metrics, val_metrics, args)
            print(f"[Best Updated] {best_ckpt_path}")

        row = {
            "epoch": epoch,
            "epoch_time_sec": round(epoch_time, 4),
            "total_elapsed_sec": round(total_elapsed, 4),

            "train_loss": round(train_metrics["loss"], 6),
            "train_mok_loss": round(train_metrics["mok_loss"], 6),
            "train_agg_loss": round(train_metrics["agg_loss"], 6),
            "train_div_loss": round(train_metrics["div_loss"], 6),
            "train_early_loss": round(train_metrics["early_loss"], 6),
            "train_balance_loss": round(train_metrics["balance_loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "train_precision": round(train_metrics["precision"], 6),
            "train_recall": round(train_metrics["recall"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "train_early_acc": round(train_metrics["early_acc"], 6),
            "train_branch_usage": str(train_metrics["branch_usage"]),

            "val_loss": round(val_metrics["loss"], 6),
            "val_mok_loss": round(val_metrics["mok_loss"], 6),
            "val_agg_loss": round(val_metrics["agg_loss"], 6),
            "val_div_loss": round(val_metrics["div_loss"], 6),
            "val_early_loss": round(val_metrics["early_loss"], 6),
            "val_balance_loss": round(val_metrics["balance_loss"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_early_acc": round(val_metrics["early_acc"], 6),
            "val_branch_usage": str(val_metrics["branch_usage"]),
        }
        append_log_csv(csv_path, row)
        print(f"[Metrics Logged] {csv_path}")

    print("\n" + "=" * 110)
    print(f"[Branch-Fix Training Done] Best Val F1 = {best_val_f1:.4f}")
    print("=" * 110)


if __name__ == "__main__":
    main()