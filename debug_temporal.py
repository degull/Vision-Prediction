# C:\Users\IIPL02\Desktop\Vision Prediction\debug_temporal.py
import os
import cv2
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.temporal_event_model import TemporalEventModel


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
            raise RuntimeError("No valid samples found for debug.")

        if verbose:
            pos = sum(1 for s in self.samples if s["label"] == 1.0)
            neg = sum(1 for s in self.samples if s["label"] == 0.0)

            print("[JAADCrossingClipDataset - Debug]")
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
# Main debug
# ============================================================
def print_tensor_info(name: str, x: torch.Tensor):
    print(f"{name:24s}: shape={tuple(x.shape)} | dtype={x.dtype} | device={x.device}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    clips_dir = r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD clips"
    annotations_dir = r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations"

    # --------------------------------------------------------
    # 1) Dataset / Loader
    # --------------------------------------------------------
    dataset = JAADCrossingClipDataset(
        clips_dir=clips_dir,
        annotations_dir=annotations_dir,
        num_frames=8,
        image_size=224,
        frame_stride=1,
        sample_stride=2,
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

    # --------------------------------------------------------
    # 2) Model
    # --------------------------------------------------------
    model = TemporalEventModel(
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

        debug_shapes=True,
    ).to(device)

    model.eval()

    # --------------------------------------------------------
    # 3) One batch forward
    # --------------------------------------------------------
    batch = next(iter(loader))
    video = batch["video"].to(device)                      # [B, T, 3, H, W]
    label = batch["label"].to(device).unsqueeze(1)        # [B, 1]

    print("\n[Input]")
    print_tensor_info("video", video)
    print_tensor_info("label", label)

    with torch.no_grad():
        out = model(video)

    print("\n[Output Keys]")
    for k in out.keys():
        print(f"- {k}")

    print("\n[Detailed Tensor Shapes]")
    print_tensor_info("logits", out["logits"])
    print_tensor_info("frame_tokens", out["frame_tokens"])
    print_tensor_info("frame_encoded_tokens", out["frame_encoded_tokens"])
    print_tensor_info("frame_pooled_bt", out["frame_pooled_bt"])
    print_tensor_info("frame_sequence", out["frame_sequence"])
    print_tensor_info("temporal_in", out["temporal_in"])
    print_tensor_info("temporal_fine", out["temporal_fine"])
    print_tensor_info("temporal_coarse", out["temporal_coarse"])
    print_tensor_info("temporal_coarse_up", out["temporal_coarse_up"])
    print_tensor_info("temporal_fused", out["temporal_fused"])
    print_tensor_info("pooled_feat", out["pooled_feat"])

    # --------------------------------------------------------
    # 4) Expected shape assertions
    # --------------------------------------------------------
    b, t, c, h, w = video.shape
    assert out["frame_tokens"].shape[0] == b
    assert out["frame_tokens"].shape[1] == t
    assert out["frame_tokens"].shape[2] == 49
    assert out["frame_tokens"].shape[3] == 768

    assert out["frame_encoded_tokens"].shape == (b * t, 49, 768)
    assert out["frame_pooled_bt"].shape == (b * t, 768)
    assert out["frame_sequence"].shape == (b, t, 768)

    assert out["temporal_in"].shape == (b, t, 768)
    assert out["temporal_fine"].shape == (b, t, 768)
    assert out["temporal_coarse_up"].shape == (b, t, 768)
    assert out["temporal_fused"].shape == (b, t, 768)

    assert out["pooled_feat"].shape == (b, 768)
    assert out["logits"].shape == (b, 1)

    print("\n[ASSERTION PASSED] All expected shapes are correct.")

    # --------------------------------------------------------
    # 5) Simple sanity stats
    # --------------------------------------------------------
    print("\n[Sanity Stats]")
    print(f"logits min/max      : {out['logits'].min().item():.6f} / {out['logits'].max().item():.6f}")
    print(f"pooled_feat mean    : {out['pooled_feat'].mean().item():.6f}")
    print(f"pooled_feat std     : {out['pooled_feat'].std().item():.6f}")
    print(f"temporal_fused mean : {out['temporal_fused'].mean().item():.6f}")
    print(f"temporal_fused std  : {out['temporal_fused'].std().item():.6f}")

    print("\n[Debug Done]")


if __name__ == "__main__":
    main()