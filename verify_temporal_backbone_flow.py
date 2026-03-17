import torch

# 실제 모델
from models.temporal_event_model import TemporalEventModel


###############################################
# Forward hook for flow verification
###############################################

def register_debug_hook(module, name):

    def hook(module, input, output):

        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        try:
            in_shape = tuple(input[0].shape)
        except:
            in_shape = "unknown"

        try:
            out_shape = tuple(out.shape)
        except:
            out_shape = "unknown"

        print("------------------------------------------------")
        print(f"[FLOW] {name}")
        print("input shape :", in_shape)
        print("output shape:", out_shape)
        print("------------------------------------------------")

    module.register_forward_hook(hook)


###############################################
# Main
###############################################

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = r"C:\Users\IIPL02\Desktop\Vision Prediction\checkpoints\temporal_crossing_mamba2scale\epoch_002_trainF1_0.9417_valF1_0.9670_valAcc_0.9569.pth"

    print("\n====================================================")
    print("VERIFY TEMPORAL BACKBONE EXECUTION FLOW")
    print("====================================================\n")

    ###############################################
    # Load model
    ###############################################

    model = TemporalEventModel()

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[Checkpoint Loaded]")
    print(checkpoint_path)
    print()

    print("[Missing Keys]", len(missing))
    print("[Unexpected Keys]", len(unexpected))
    print()

    model = model.to(device)
    model.eval()

    ###############################################
    # Register hooks
    ###############################################

    print("Registering module hooks...\n")

    if hasattr(model, "backbone"):
        register_debug_hook(model.backbone, "Spatial Backbone (ConvNeXtV2)")

    if hasattr(model, "frame_encoder"):
        register_debug_hook(model.frame_encoder, "Transformer + Volterra Frame Encoder")

    if hasattr(model, "temporal_encoder"):
        register_debug_hook(model.temporal_encoder, "2-Scale Temporal Mamba Encoder")

    if hasattr(model, "event_head"):
        register_debug_hook(model.event_head, "Event Prediction Head")

    ###############################################
    # Create dummy video clip
    ###############################################

    # (batch, frames, channels, H, W)
    frames = torch.randn(1, 8, 3, 224, 224).to(device)

    print("\n====================================================")
    print("DUMMY INPUT")
    print("====================================================\n")

    print("frames shape:", frames.shape)

    ###############################################
    # Forward pass
    ###############################################

    print("\n====================================================")
    print("FORWARD EXECUTION")
    print("====================================================\n")

    with torch.no_grad():
        output = model(frames)

    ###############################################
    # Final output
    ###############################################

    print("\n====================================================")
    print("FINAL OUTPUT")
    print("====================================================\n")

    print("model output type:", type(output))

    if isinstance(output, dict):

        print("\nOutput keys:", output.keys())

        for k, v in output.items():

            if hasattr(v, "shape"):
                print(f"{k} shape:", tuple(v.shape))
            else:
                print(f"{k}:", v)

    else:

        print("output shape:", tuple(output.shape))


###############################################

if __name__ == "__main__":
    main()