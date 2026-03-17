# C:\Users\IIPL02\Desktop\Vision Prediction\debug_backbone.py

import torch
from torch.utils.data import DataLoader

from datasets.jaad_video_dataset import JAADVideoDataset
from models.backbone import ConvNeXtV2Backbone


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clips_dir = r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD clips"

    dataset = JAADVideoDataset(
        clips_dir=clips_dir,
        num_frames=8,
        image_size=224,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------
    # Backbone (token mode)
    # --------------------------------
    model = ConvNeXtV2Backbone(
        model_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=True,
        freeze=True,
        output_mode="tokens",   # 🔴 중요
    ).to(device)

    model.eval()

    batch = next(iter(loader))
    video = batch["video"].to(device)   # [B, T, 3, 224, 224]

    print("input video shape:", video.shape)

    with torch.no_grad():
        feat = model(video)

    print("backbone token shape:", feat.shape)  
    # expected [B, T, 49, 768]

    # sanity check
    B, T, N, C = feat.shape

    print("\n[Details]")
    print("B (batch)  :", B)
    print("T (frames) :", T)
    print("N (tokens) :", N)
    print("C (dim)    :", C)

    assert N == 49, "ConvNeXt 224 input should produce 7x7 tokens -> 49"
    assert C == 768

    print("\n[ASSERTION PASSED]")


if __name__ == "__main__":
    main()