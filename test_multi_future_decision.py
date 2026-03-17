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
# Dataset
# ============================================================
class JAADCrossingDecisionDataset(Dataset):
    """
    Per-pedestrian clip dataset.

    One sample:
        (video clip, target pedestrian track, end_frame)

    Targets:
        crossing_label : 0/1 at end_frame for this pedestrian track
        early_label    : 1 if ANY decision_point in the same video occurs
                         within next early_horizon frames
        early_valid    : 1 if the video has at least one valid decision_point

    Note:
        This uses video-level weak early-decision supervision because
        pedestrian IDs in annotations/*.xml and annotations_attributes/*.xml
        may not align 1:1.
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
def min_of_k_bce_loss(branch_logits, labels):
    labels_exp = labels.unsqueeze(1).expand(-1, branch_logits.size(1), -1)
    per_branch_loss = F.binary_cross_entropy_with_logits(
        branch_logits,
        labels_exp,
        reduction="none"
    ).squeeze(-1)

    min_loss, best_branch_idx = per_branch_loss.min(dim=1)
    loss = min_loss.mean()
    return loss, best_branch_idx, min_loss


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


# ============================================================
# Evaluation
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


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    lambda_mok=1.0,
    lambda_agg=1.0,
    lambda_div=0.05,
    lambda_early=0.5,
    log_interval=50,
    prediction_csv_path=None,
):
    model.eval()

    total_loss_meter = AverageMeter()
    mok_loss_meter = AverageMeter()
    agg_loss_meter = AverageMeter()
    div_loss_meter = AverageMeter()
    early_loss_meter = AverageMeter()

    all_probs = []
    all_labels = []
    all_early_logits = []
    all_early_labels = []
    all_early_valid = []

    branch_usage = torch.zeros(model.num_futures, dtype=torch.long)

    sample_rows = []

    start_time = time.time()
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        video = batch["video"].to(device, non_blocking=True)
        crossing_labels = batch["crossing_label"].to(device, non_blocking=True).unsqueeze(1)
        early_labels = batch["early_label"].to(device, non_blocking=True).unsqueeze(1)
        early_valid = batch["early_valid"].to(device, non_blocking=True).unsqueeze(1)

        out = model(video)

        branch_logits = out["branch_logits"]
        agg_prob = out["agg_prob"]
        future_feats = out["future_feats"]
        early_logit = out["early_logit"]

        mok_loss, best_branch_idx, per_sample_min_loss = min_of_k_bce_loss(branch_logits, crossing_labels)
        agg_loss = aggregation_bce_loss(agg_prob, crossing_labels)
        div_loss = diversity_loss_cosine(future_feats)
        early_loss = masked_bce_with_logits(early_logit, early_labels, early_valid)

        total_loss = (
            lambda_mok * mok_loss +
            lambda_agg * agg_loss +
            lambda_div * div_loss +
            lambda_early * early_loss
        )

        bs = video.size(0)

        total_loss_meter.update(total_loss.item(), bs)
        mok_loss_meter.update(mok_loss.item(), bs)
        agg_loss_meter.update(agg_loss.item(), bs)
        div_loss_meter.update(div_loss.item(), bs)
        early_loss_meter.update(early_loss.item(), bs)

        all_probs.append(agg_prob.detach().cpu())
        all_labels.append(crossing_labels.detach().cpu())
        all_early_logits.append(early_logit.detach().cpu())
        all_early_labels.append(early_labels.detach().cpu())
        all_early_valid.append(early_valid.detach().cpu())

        branch_probs = torch.sigmoid(branch_logits).detach().cpu().squeeze(-1)
        agg_prob_cpu = agg_prob.detach().cpu().squeeze(-1)
        crossing_labels_cpu = crossing_labels.detach().cpu().squeeze(-1)
        early_prob_cpu = torch.sigmoid(early_logit.detach().cpu()).squeeze(-1)
        early_labels_cpu = early_labels.detach().cpu().squeeze(-1)
        early_valid_cpu = early_valid.detach().cpu().squeeze(-1)
        best_branch_idx_cpu = best_branch_idx.detach().cpu()

        for idx in best_branch_idx_cpu.tolist():
            branch_usage[idx] += 1

        for i in range(bs):
            row = {
                "video_path": batch["video_path"][i],
                "ped_id": batch["ped_id"][i],
                "old_id": batch["old_id"][i] if batch["old_id"][i] is not None else "",
                "track_id": batch["track_id"][i],
                "end_frame": int(batch["end_frame"][i]),
                "crossing_label": float(crossing_labels_cpu[i].item()),
                "agg_prob": float(agg_prob_cpu[i].item()),
                "agg_pred": int(agg_prob_cpu[i].item() >= 0.5),
                "best_branch_idx": int(best_branch_idx_cpu[i].item()),
                "branch0_prob": float(branch_probs[i, 0].item()) if branch_probs.size(1) > 0 else -1.0,
                "branch1_prob": float(branch_probs[i, 1].item()) if branch_probs.size(1) > 1 else -1.0,
                "branch2_prob": float(branch_probs[i, 2].item()) if branch_probs.size(1) > 2 else -1.0,
                "early_valid": float(early_valid_cpu[i].item()),
                "early_label": float(early_labels_cpu[i].item()),
                "early_prob": float(early_prob_cpu[i].item()),
                "early_pred": int(early_prob_cpu[i].item() >= 0.5),
                "per_sample_min_mok_loss": float(per_sample_min_loss.detach().cpu()[i].item()),
            }
            sample_rows.append(row)

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

            print(
                f"\r[Test] Batch {batch_idx:04d}/{num_batches:04d} | "
                f"Loss {total_loss_meter.avg:.4f} | "
                f"MoK {mok_loss_meter.avg:.4f} | "
                f"Agg {agg_loss_meter.avg:.4f} | "
                f"Div {div_loss_meter.avg:.4f} | "
                f"Early {early_loss_meter.avg:.4f} | "
                f"CrossF1 {batch_metrics['f1']:.4f} | "
                f"EarlyAcc {batch_early_acc:.4f} | "
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
    metrics["early_acc"] = early_acc
    metrics["branch_usage"] = branch_usage.tolist()

    if prediction_csv_path is not None:
        ensure_dir(os.path.dirname(prediction_csv_path))
        with open(prediction_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)
        print(f"[Prediction CSV Saved] {prediction_csv_path}")

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_decision_branchfix\best_branchfix_epoch_003_valF1_0.8818_valEarly_0.8838.pth",
        help="checkpoint path"
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

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "all"],
        help="Which split to evaluate"
    )

    parser.add_argument("--num_futures", type=int, default=3)
    parser.add_argument("--future_dim", type=int, default=256)

    parser.add_argument("--lambda_mok", type=float, default=1.0)
    parser.add_argument("--lambda_agg", type=float, default=1.0)
    parser.add_argument("--lambda_div", type=float, default=0.05)
    parser.add_argument("--lambda_early", type=float, default=0.5)

    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\multi_future_decision_test"
    )
    parser.add_argument(
        "--prediction_csv_name",
        type=str,
        default="predictions.csv"
    )

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    print(f"[Load Checkpoint] {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    ckpt_args = ckpt.get("args", {})
    if len(ckpt_args) > 0:
        print("[Checkpoint Args Found]")
        for k in [
            "num_frames", "image_size", "frame_stride", "sample_stride", "early_horizon",
            "num_futures", "future_dim", "val_ratio"
        ]:
            if k in ckpt_args:
                print(f"  {k}: {ckpt_args[k]}")
        print()

    num_frames = ckpt_args.get("num_frames", args.num_frames)
    image_size = ckpt_args.get("image_size", args.image_size)
    frame_stride = ckpt_args.get("frame_stride", args.frame_stride)
    sample_stride = ckpt_args.get("sample_stride", args.sample_stride)
    early_horizon = ckpt_args.get("early_horizon", args.early_horizon)
    num_futures = ckpt_args.get("num_futures", args.num_futures)
    future_dim = ckpt_args.get("future_dim", args.future_dim)
    val_ratio = ckpt_args.get("val_ratio", args.val_ratio)

    full_dataset = JAADCrossingDecisionDataset(
        clips_dir=args.clips_dir,
        annotations_dir=args.annotations_dir,
        attributes_dir=args.attributes_dir,
        num_frames=num_frames,
        image_size=image_size,
        frame_stride=frame_stride,
        sample_stride=sample_stride,
        early_horizon=early_horizon,
        verbose=True,
    )

    if args.split == "all":
        eval_set = full_dataset
        print(f"[Split] all={len(eval_set)}")
    else:
        val_size = max(1, int(len(full_dataset) * val_ratio))
        train_size = len(full_dataset) - val_size
        if train_size <= 0:
            raise RuntimeError("Dataset too small after split.")

        train_set, val_set = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        if args.split == "train":
            eval_set = train_set
        else:
            eval_set = val_set

        print(f"[Split] train={len(train_set)} | val={len(val_set)} | eval={args.split}:{len(eval_set)}")

    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiFutureCrossingDecisionModel(
        backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
        backbone_dim=768,
        temporal_dim=768,
        future_dim=future_dim,
        num_futures=num_futures,
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
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    prediction_csv_path = os.path.join(args.save_dir, args.prediction_csv_name)

    start = time.time()
    metrics = evaluate(
        model=model,
        loader=eval_loader,
        device=device,
        lambda_mok=args.lambda_mok,
        lambda_agg=args.lambda_agg,
        lambda_div=args.lambda_div,
        lambda_early=args.lambda_early,
        log_interval=args.log_interval,
        prediction_csv_path=prediction_csv_path,
    )
    elapsed = time.time() - start

    print("\n" + "=" * 110)
    print(f"[Evaluation Done] Time {format_seconds(elapsed)}")
    print(
        f"[Result] "
        f"Loss {metrics['loss']:.4f} | "
        f"MoK {metrics['mok_loss']:.4f} | "
        f"Agg {metrics['agg_loss']:.4f} | "
        f"Div {metrics['div_loss']:.4f} | "
        f"Early {metrics['early_loss']:.4f} | "
        f"CrossAcc {metrics['acc']:.4f} | "
        f"Precision {metrics['precision']:.4f} | "
        f"Recall {metrics['recall']:.4f} | "
        f"CrossF1 {metrics['f1']:.4f} | "
        f"EarlyAcc {metrics['early_acc']:.4f} | "
        f"BranchUsage {metrics['branch_usage']}"
    )
    print("=" * 110)

    summary_txt = os.path.join(args.save_dir, "test_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"split: {args.split}\n")
        f.write(f"time_sec: {elapsed:.4f}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"[Summary Saved] {summary_txt}")


if __name__ == "__main__":
    main()

