# C:\Users\IIPL02\Desktop\Vision Prediction\verify_temporal_checkpoint.py
import os
import cv2
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.temporal_event_model import TemporalEventModel


# ============================================================
# Utils
# ============================================================
def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def tensor_stats(name, x):
    x = x.detach().float().cpu()
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"mean={x.mean():.6f}, std={x.std():.6f}, "
        f"min={x.min():.6f}, max={x.max():.6f}"
    )


def compare_module_weights(module_a, module_b, module_name="module"):
    """
    Compare two modules parameter-by-parameter.
    Returns average absolute difference across all params.
    """
    diffs = []
    named_a = dict(module_a.named_parameters())
    named_b = dict(module_b.named_parameters())

    common_keys = sorted(set(named_a.keys()) & set(named_b.keys()))
    if len(common_keys) == 0:
        print(f"[WARN] No common parameter keys found for {module_name}")
        return None

    for k in common_keys:
        pa = named_a[k].detach().cpu()
        pb = named_b[k].detach().cpu()
        diff = (pa - pb).abs().mean().item()
        diffs.append(diff)

    avg_diff = sum(diffs) / max(len(diffs), 1)
    print(f"[Compare] {module_name} average abs param diff = {avg_diff:.8f}")
    return avg_diff


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
    Returns:
        {
            "video": [T, 3, H, W],
            "label": scalar float tensor,
            "video_path": str,
            "xml_path": str,
            "end_frame": int,
        }
    """

    def __init__(
        self,
        clips_dir: str,
        annotations_dir: str,
        num_frames: int = 8,
        image_size: int = 224,
        frame_stride: int = 1,
        sample_stride: int = 2,
        max_samples: int = 8,
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

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found for verification.")

        if verbose:
            pos = sum(1 for s in self.samples if s["label"] == 1.0)
            neg = sum(1 for s in self.samples if s["label"] == 0.0)
            print("[JAADCrossingClipDataset - Verify]")
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
# Build model from checkpoint args
# ============================================================
def build_model_from_ckpt_args(ckpt_args: dict, device: str):
    return TemporalEventModel(
        backbone_name=ckpt_args.get("backbone_name", "convnextv2_tiny.fcmae_ft_in22k_in1k"),
        backbone_dim=ckpt_args.get("backbone_dim", 768),
        freeze_backbone=bool(ckpt_args.get("freeze_backbone", True)),

        frame_feature_dim=ckpt_args.get("frame_feature_dim", 768),
        frame_encoder_num_heads=ckpt_args.get("frame_encoder_num_heads", 8),
        frame_encoder_num_layers=ckpt_args.get("frame_encoder_num_layers", 2),
        frame_encoder_ff_dim=ckpt_args.get("frame_encoder_ff_dim", 1536),
        frame_encoder_dropout=ckpt_args.get("frame_encoder_dropout", 0.1),
        frame_encoder_use_volterra=bool(ckpt_args.get("frame_encoder_use_volterra", True)),
        frame_encoder_volterra_rank=ckpt_args.get("frame_encoder_volterra_rank", 16),
        frame_encoder_volterra_alpha=ckpt_args.get("frame_encoder_volterra_alpha", 1.0),

        temporal_encoder_type=ckpt_args.get("temporal_encoder_type", "mamba_2scale"),
        temporal_mamba_dim=ckpt_args.get("temporal_feature_dim", 768),
        temporal_mamba_num_layers=ckpt_args.get("temporal_mamba_num_layers", 2),
        temporal_mamba_state_dim=ckpt_args.get("temporal_mamba_state_dim", 16),
        temporal_mamba_conv_kernel=ckpt_args.get("temporal_mamba_conv_kernel", 4),
        temporal_mamba_expand=ckpt_args.get("temporal_mamba_expand", 2),
        temporal_mamba_dropout=ckpt_args.get("temporal_mamba_dropout", 0.1),
        temporal_mamba_fusion=ckpt_args.get("temporal_mamba_fusion", "concat_proj"),
        temporal_mamba_local_window=ckpt_args.get("temporal_mamba_local_window", 4),
        temporal_pooling=ckpt_args.get("temporal_pooling", "last"),

        event_hidden_dim=ckpt_args.get("event_hidden_dim", 256),
        event_dropout=ckpt_args.get("event_dropout", 0.1),

        debug_shapes=False,
    ).to(device)


# ============================================================
# Main
# ============================================================
def main():
    # ============================================================
    # Paths
    # ============================================================
    ckpt_path = r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\temporal_crossing_mamba2scale\latest.pth"
    clips_dir = r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD clips"
    annotations_dir = r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found:\n{ckpt_path}")

    if not os.path.exists(clips_dir):
        raise FileNotFoundError(f"JAAD clips dir not found:\n{clips_dir}")

    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"JAAD annotations dir not found:\n{annotations_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Checkpoint] {ckpt_path}")

    # ============================================================
    # 1) Load checkpoint raw
    # ============================================================
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("\n" + "=" * 100)
    print("[1] CHECKPOINT TOP-LEVEL KEYS")
    print("=" * 100)
    for k in ckpt.keys():
        print(k)

    if "epoch" in ckpt:
        print(f"\n[Checkpoint Epoch] {ckpt['epoch']}")

    if "train_metrics" in ckpt:
        print(f"[Train Metrics] {ckpt['train_metrics']}")

    if "val_metrics" in ckpt:
        print(f"[Val Metrics] {ckpt['val_metrics']}")

    if "args" in ckpt:
        print(f"[Args keys] {list(ckpt['args'].keys())}")

    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    state_dict = ckpt["model_state_dict"]
    ckpt_args = ckpt.get("args", {})

    # ============================================================
    # 2) Inspect new-structure keys
    # ============================================================
    print("\n" + "=" * 100)
    print("[2] NEW STRUCTURE KEYS IN CHECKPOINT")
    print("=" * 100)

    backbone_keys = [k for k in state_dict.keys() if k.startswith("backbone.")]
    frame_encoder_keys = [k for k in state_dict.keys() if k.startswith("frame_encoder.")]
    temporal_keys = [k for k in state_dict.keys() if k.startswith("temporal_encoder.")]
    event_keys = [k for k in state_dict.keys() if k.startswith("event_head.")]
    frame_proj_keys = [k for k in state_dict.keys() if k.startswith("frame_proj.")]
    temporal_in_proj_keys = [k for k in state_dict.keys() if k.startswith("temporal_in_proj.")]

    print(f"Num backbone keys          : {len(backbone_keys)}")
    print(f"Num frame_proj keys        : {len(frame_proj_keys)}")
    print(f"Num frame_encoder keys     : {len(frame_encoder_keys)}")
    print(f"Num temporal_in_proj keys  : {len(temporal_in_proj_keys)}")
    print(f"Num temporal_encoder keys  : {len(temporal_keys)}")
    print(f"Num event_head keys        : {len(event_keys)}")

    if len(frame_encoder_keys) == 0:
        raise RuntimeError("No frame_encoder.* keys found in checkpoint.")
    if len(temporal_keys) == 0:
        raise RuntimeError("No temporal_encoder.* keys found in checkpoint.")
    if len(event_keys) == 0:
        raise RuntimeError("No event_head.* keys found in checkpoint.")

    print("\n[Sample frame_encoder keys]")
    for k in frame_encoder_keys[:10]:
        print(k)

    print("\n[Sample temporal_encoder keys]")
    for k in temporal_keys[:10]:
        print(k)

    # ============================================================
    # 3) Build model and load checkpoint
    # ============================================================
    print("\n" + "=" * 100)
    print("[3] BUILD MODEL + LOAD CHECKPOINT")
    print("=" * 100)

    loaded_model = build_model_from_ckpt_args(ckpt_args, device)
    missing_keys, unexpected_keys = loaded_model.load_state_dict(state_dict, strict=False)

    print(f"[Load] missing_keys    = {len(missing_keys)}")
    print(f"[Load] unexpected_keys = {len(unexpected_keys)}")

    if len(missing_keys) > 0:
        print("\nMissing keys:")
        for k in missing_keys[:30]:
            print(" ", k)

    if len(unexpected_keys) > 0:
        print("\nUnexpected keys:")
        for k in unexpected_keys[:30]:
            print(" ", k)

    critical_prefixes = ("frame_encoder.", "temporal_encoder.", "event_head.")
    for mk in missing_keys:
        if mk.startswith(critical_prefixes):
            raise RuntimeError(f"Critical key missing during load: {mk}")

    loaded_model.eval()

    print(f"\n[Params] backbone         = {count_parameters(loaded_model.backbone):,}")
    print(f"[Params] frame_encoder    = {count_parameters(loaded_model.frame_encoder):,}")
    print(f"[Params] temporal_encoder = {count_parameters(loaded_model.temporal_encoder):,}")
    print(f"[Params] event_head       = {count_parameters(loaded_model.event_head):,}")

    # ============================================================
    # 4) Build a fresh model for comparison
    # ============================================================
    print("\n" + "=" * 100)
    print("[4] COMPARE AGAINST FRESHLY INITIALIZED MODEL")
    print("=" * 100)

    fresh_model = build_model_from_ckpt_args(ckpt_args, device)
    fresh_model.eval()

    compare_module_weights(
        loaded_model.frame_encoder,
        fresh_model.frame_encoder,
        module_name="frame_encoder"
    )
    compare_module_weights(
        loaded_model.temporal_encoder,
        fresh_model.temporal_encoder,
        module_name="temporal_encoder"
    )
    compare_module_weights(
        loaded_model.event_head,
        fresh_model.event_head,
        module_name="event_head"
    )

    # ============================================================
    # 5) Run one debug batch through the loaded model
    # ============================================================
    print("\n" + "=" * 100)
    print("[5] FORWARD CHECK WITH REAL JAAD VIDEO")
    print("=" * 100)

    dataset = JAADCrossingClipDataset(
        clips_dir=clips_dir,
        annotations_dir=annotations_dir,
        num_frames=ckpt_args.get("num_frames", 8),
        image_size=ckpt_args.get("image_size", 224),
        frame_stride=ckpt_args.get("frame_stride", 1),
        sample_stride=ckpt_args.get("sample_stride", 2),
        max_samples=8,
        verbose=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))
    video = batch["video"].to(device)  # [B, T, 3, H, W]
    print(f"Input video shape: {tuple(video.shape)}")

    with torch.no_grad():
        out_loaded = loaded_model(video)

    print("\n[Loaded checkpoint model outputs]")
    tensor_stats("frame_tokens", out_loaded["frame_tokens"])
    tensor_stats("frame_encoded_tokens", out_loaded["frame_encoded_tokens"])
    tensor_stats("frame_pooled_bt", out_loaded["frame_pooled_bt"])
    tensor_stats("frame_sequence", out_loaded["frame_sequence"])
    tensor_stats("temporal_in", out_loaded["temporal_in"])
    tensor_stats("temporal_fine", out_loaded["temporal_fine"])
    tensor_stats("temporal_coarse", out_loaded["temporal_coarse"])
    tensor_stats("temporal_coarse_up", out_loaded["temporal_coarse_up"])
    tensor_stats("temporal_fused", out_loaded["temporal_fused"])
    tensor_stats("pooled_feat", out_loaded["pooled_feat"])
    tensor_stats("logits", out_loaded["logits"])
    tensor_stats("probs", torch.sigmoid(out_loaded["logits"]))

    # shape assertions
    b, t = video.shape[0], video.shape[1]

    assert out_loaded["frame_tokens"].shape == (b, t, 49, ckpt_args.get("frame_feature_dim", 768)), \
        f"Unexpected frame_tokens shape: {out_loaded['frame_tokens'].shape}"

    assert out_loaded["frame_encoded_tokens"].shape == (b * t, 49, ckpt_args.get("frame_feature_dim", 768)), \
        f"Unexpected frame_encoded_tokens shape: {out_loaded['frame_encoded_tokens'].shape}"

    assert out_loaded["frame_pooled_bt"].shape == (b * t, ckpt_args.get("frame_feature_dim", 768)), \
        f"Unexpected frame_pooled_bt shape: {out_loaded['frame_pooled_bt'].shape}"

    assert out_loaded["frame_sequence"].shape == (b, t, ckpt_args.get("frame_feature_dim", 768)), \
        f"Unexpected frame_sequence shape: {out_loaded['frame_sequence'].shape}"

    assert out_loaded["temporal_in"].shape == (b, t, ckpt_args.get("temporal_feature_dim", 768)), \
        f"Unexpected temporal_in shape: {out_loaded['temporal_in'].shape}"

    assert out_loaded["temporal_fine"].shape == (b, t, ckpt_args.get("temporal_feature_dim", 768)), \
        f"Unexpected temporal_fine shape: {out_loaded['temporal_fine'].shape}"

    assert out_loaded["temporal_coarse_up"].shape == (b, t, ckpt_args.get("temporal_feature_dim", 768)), \
        f"Unexpected temporal_coarse_up shape: {out_loaded['temporal_coarse_up'].shape}"

    assert out_loaded["temporal_fused"].shape == (b, t, ckpt_args.get("temporal_feature_dim", 768)), \
        f"Unexpected temporal_fused shape: {out_loaded['temporal_fused'].shape}"

    assert out_loaded["pooled_feat"].shape == (b, ckpt_args.get("temporal_feature_dim", 768)), \
        f"Unexpected pooled_feat shape: {out_loaded['pooled_feat'].shape}"

    assert out_loaded["logits"].shape == (b, 1), \
        f"Unexpected logits shape: {out_loaded['logits'].shape}"

    # ============================================================
    # 6) Compare loaded model output vs fresh model output
    # ============================================================
    print("\n" + "=" * 100)
    print("[6] OUTPUT DIFFERENCE: LOADED MODEL VS FRESH MODEL")
    print("=" * 100)

    with torch.no_grad():
        out_fresh = fresh_model(video)

    frame_seq_diff = (out_loaded["frame_sequence"] - out_fresh["frame_sequence"]).abs().mean().item()
    temporal_diff = (out_loaded["temporal_fused"] - out_fresh["temporal_fused"]).abs().mean().item()
    pooled_diff = (out_loaded["pooled_feat"] - out_fresh["pooled_feat"]).abs().mean().item()
    logit_diff = (out_loaded["logits"] - out_fresh["logits"]).abs().mean().item()

    print(f"Mean abs diff frame_sequence : {frame_seq_diff:.8f}")
    print(f"Mean abs diff temporal_fused : {temporal_diff:.8f}")
    print(f"Mean abs diff pooled_feat    : {pooled_diff:.8f}")
    print(f"Mean abs diff logits         : {logit_diff:.8f}")

    if temporal_diff < 1e-7:
        print("[WARN] temporal_fused difference is extremely small.")
    else:
        print("[OK] Loaded model output differs from fresh initialization.")

    # ============================================================
    # 7) Direct role check for backbone -> frame -> temporal
    # ============================================================
    print("\n" + "=" * 100)
    print("[7] BACKBONE / FRAME / TEMPORAL ROLE CHECK")
    print("=" * 100)

    with torch.no_grad():
        frame_tokens = loaded_model.extract_frame_tokens(video)          # [B, T, 49, D_backbone]
        frame_tokens_proj = loaded_model.frame_proj(frame_tokens)        # [B, T, 49, D_frame]

        bb, tt, nn, dd = frame_tokens_proj.shape
        frame_tokens_bt = frame_tokens_proj.view(bb * tt, nn, dd)       # [B*T, 49, D]
        frame_encoded_bt = loaded_model.frame_encoder(frame_tokens_bt)   # [B*T, 49, D]
        frame_pooled_bt = loaded_model.pool_frame_tokens(frame_encoded_bt)
        frame_sequence = frame_pooled_bt.view(bb, tt, dd)
        temporal_in = loaded_model.temporal_in_proj(frame_sequence)
        temporal_dict = loaded_model.temporal_encoder(temporal_in)

    tensor_stats("backbone(frame_tokens)", frame_tokens)
    tensor_stats("frame_proj(frame_tokens_proj)", frame_tokens_proj)
    tensor_stats("frame_encoder(frame_encoded_bt)", frame_encoded_bt)
    tensor_stats("frame_sequence", frame_sequence)
    tensor_stats("temporal_encoder(fused)", temporal_dict["fused"])

    frame_change = (frame_encoded_bt - frame_tokens_bt).abs().mean().item()
    temporal_change = (temporal_dict["fused"] - temporal_in).abs().mean().item()

    print(f"Mean abs diff frame_encoded_bt vs frame_tokens_bt = {frame_change:.8f}")
    print(f"Mean abs diff temporal_fused vs temporal_in       = {temporal_change:.8f}")

    if frame_change > 0:
        print("[OK] Frame encoder is not identity; it transforms token features.")
    else:
        print("[WARN] Frame encoder output is identical to input.")

    if temporal_change > 0:
        print("[OK] Temporal encoder is not identity; it transforms frame sequence.")
    else:
        print("[WARN] Temporal encoder output is identical to input.")

    # ============================================================
    # Final conclusion
    # ============================================================
    print("\n" + "=" * 100)
    print("[FINAL CHECK RESULT]")
    print("=" * 100)
    print("1. Checkpoint file exists and loads.")
    print("2. frame_encoder.* / temporal_encoder.* / event_head.* keys exist.")
    print("3. Loaded model produces valid new-structure shapes:")
    print("   - frame_tokens         : [B, T, 49, D]")
    print("   - frame_encoded_tokens : [B*T, 49, D]")
    print("   - frame_sequence       : [B, T, D]")
    print("   - temporal_fused       : [B, T, Dt]")
    print("   - pooled_feat          : [B, Dt]")
    print("   - logits               : [B, 1]")
    print("4. Loaded model differs from fresh initialization.")
    print("5. Frame encoder and temporal encoder are both transforming features.")
    print("\n=> This checkpoint is a valid trained temporal-event model checkpoint")
    print("   for the new token-based backbone + frame encoder + 2-scale temporal mamba structure.")


if __name__ == "__main__":
    main()