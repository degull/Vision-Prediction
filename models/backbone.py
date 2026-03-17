# C:\Users\IIPL02\Desktop\Vision Prediction\models\backbone.py
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXtV2 backbone for video clips.

    Supported output modes
    ----------------------
    1) pooled
       input : [B, T, 3, H, W]
       output: [B, T, C]

    2) featmap
       input : [B, T, 3, H, W]
       output: [B, T, C, H', W']

    3) tokens
       input : [B, T, 3, H, W]
       output: [B, T, N, C]
               where N = H' * W'
    """

    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        freeze: bool = True,
        output_mode: str = "pooled",   # "pooled" | "featmap" | "tokens"
    ):
        super().__init__()

        if timm is None:
            raise ImportError("Please install timm: pip install timm")

        valid_modes = ["pooled", "featmap", "tokens"]
        if output_mode not in valid_modes:
            raise ValueError(
                f"Unsupported output_mode: {output_mode}. "
                f"Choose from {valid_modes}"
            )

        self.model_name = model_name
        self.output_mode = output_mode

        # ----------------------------------------------------
        # pooled mode:
        #   backbone(x) -> [B*T, C]
        #
        # featmap/tokens mode:
        #   backbone.forward_features(x) -> [B*T, C, H', W']
        # ----------------------------------------------------
        if output_mode == "pooled":
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
            )

        self.out_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, 3, H, W]

        Returns:
            pooled  -> [B, T, C]
            featmap -> [B, T, C, H', W']
            tokens  -> [B, T, N, C]
        """
        if video.dim() != 5:
            raise ValueError(
                f"Expected video shape [B, T, 3, H, W], but got {tuple(video.shape)}"
            )

        B, T, C, H, W = video.shape
        x = video.view(B * T, C, H, W)  # [B*T, 3, H, W]

        # ----------------------------------------------------
        # pooled vector output
        # ----------------------------------------------------
        if self.output_mode == "pooled":
            feat = self.backbone(x)            # [B*T, C]
            if feat.dim() != 2:
                raise RuntimeError(
                    f"Expected pooled output [B*T, C], but got {tuple(feat.shape)}"
                )
            feat = feat.view(B, T, -1)         # [B, T, C]
            return feat

        # ----------------------------------------------------
        # feature map output
        # ----------------------------------------------------
        feat = self.backbone.forward_features(x)   # expected [B*T, C, H', W']

        if feat.dim() != 4:
            raise RuntimeError(
                f"Expected feature map [B*T, C, H', W'], but got {tuple(feat.shape)}"
            )

        _, C2, H2, W2 = feat.shape
        feat = feat.view(B, T, C2, H2, W2)         # [B, T, C, H', W']

        if self.output_mode == "featmap":
            return feat

        # ----------------------------------------------------
        # tokens output
        # [B, T, C, H', W'] -> [B, T, N, C]
        # ----------------------------------------------------
        feat = feat.flatten(3)                         # [B, T, C, N]
        feat = feat.transpose(-1, -2).contiguous()    # [B, T, N, C]
        return feat


if __name__ == "__main__":
    # Simple shape test
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_video = torch.randn(2, 8, 3, 224, 224).to(device)

    for mode in ["pooled", "featmap", "tokens"]:
        print(f"\n[Test] output_mode={mode}")
        model = ConvNeXtV2Backbone(
            model_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=False,
            freeze=True,
            output_mode=mode,
        ).to(device)

        with torch.no_grad():
            out = model(dummy_video)

        print(f"output shape: {tuple(out.shape)}")