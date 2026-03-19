import os
import re
import random
import inspect
import importlib
import argparse
from collections import defaultdict

import torch
from torch.utils.data import random_split


# ============================================================
# Utility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset Resolver
# ============================================================
def resolve_dataset_class():
    candidates = [
        ("datasets.jaad_video_dataset", ["JAADVideoDataset"]),
        ("jaad_video_dataset", ["JAADVideoDataset"]),
    ]

    for module_name, class_names in candidates:
        try:
            module = importlib.import_module(module_name)
            for cls_name in class_names:
                if hasattr(module, cls_name):
                    print(f"[Dataset Resolved] module={module_name} | class={cls_name}")
                    return getattr(module, cls_name)
        except Exception:
            continue

    raise ImportError("Dataset class not found.")


def build_dataset(dataset_cls, args):
    sig = inspect.signature(dataset_cls.__init__)
    supported = set(sig.parameters.keys())

    candidate_kwargs = {
        "clips_dir": args.clips_dir,
        "annotations_dir": args.annotations_dir,
        "attributes_dir": args.attributes_dir,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "frame_stride": args.frame_stride,
        "sample_stride": args.sample_stride,
        "early_horizon": args.early_horizon,
        "verbose": args.dataset_verbose,
    }

    final_kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}

    print("[Dataset Init Args]")
    for k, v in final_kwargs.items():
        print(f"  {k}: {v}")
    print()

    return dataset_cls(**final_kwargs)


# ============================================================
# Preview paths (🔥 추가된 부분)
# ============================================================
def preview_paths(dataset, num_samples=20):
    print("[Video Path Preview]")
    shown = 0
    for i in range(min(len(dataset), num_samples)):
        sample = dataset[i]
        if "video_path" in sample:
            print(f"{i:03d}: {sample['video_path']}")
            shown += 1
    if shown == 0:
        print("No video_path found.")
    print()


# ============================================================
# Key detection
# ============================================================
def infer_keys(sample):
    video_key = "video" if "video" in sample else None
    label_key = "crossing_label" if "crossing_label" in sample else None
    path_key = "video_path" if "video_path" in sample else None

    print("[Detected Keys]")
    print(f"  video_key    : {video_key}")
    print(f"  label_key    : {label_key}")
    print(f"  path_key     : {path_key}")
    print()

    return {
        "video_key": video_key,
        "label_key": label_key,
        "path_key": path_key,
    }


# ============================================================
# Metadata builder
# ============================================================
def normalize_path(p):
    if p is None:
        return None
    return os.path.normpath(str(p)).replace("\\", "/")


def build_meta(sample, keys, idx):
    path = normalize_path(sample.get(keys["path_key"]))
    label = sample.get(keys["label_key"])

    if isinstance(label, torch.Tensor):
        label = label.item()

    return {
        "idx": idx,
        "path": path,
        "label": int(label) if label is not None else None,
    }


# ============================================================
# Summary
# ============================================================
def summarize(name, metas):
    pos = sum(1 for m in metas if m["label"] == 1)
    neg = sum(1 for m in metas if m["label"] == 0)

    print(f"[{name}]")
    print(f"  samples : {len(metas)}")
    print(f"  pos/neg : {pos}/{neg}")
    print()


# ============================================================
# Overlap check
# ============================================================
def check_overlap(train_vals, val_vals, name):
    train_set = set(train_vals)
    val_set = set(val_vals)

    overlap = train_set & val_set

    print(f"[Leakage Check: {name}]")
    print(f"  overlap: {len(overlap)}")

    if overlap:
        print("  examples:")
        for x in list(overlap)[:10]:
            print(f"    {x}")
    print()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--attributes_dir", type=str, required=True)

    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--early_horizon", type=int, default=30)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--dataset_verbose", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    dataset_cls = resolve_dataset_class()
    dataset = build_dataset(dataset_cls, args)

    print(f"[Dataset Size] {len(dataset)}\n")

    # 🔥 sample 확인
    sample0 = dataset[0]
    print(f"[Sample Keys] {list(sample0.keys())}\n")

    # 🔥 path preview 추가
    preview_paths(dataset, num_samples=30)

    keys = infer_keys(sample0)

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"[Split] train={len(train_set)} | val={len(val_set)}\n")

    train_metas = [build_meta(dataset[i], keys, i) for i in train_set.indices]
    val_metas = [build_meta(dataset[i], keys, i) for i in val_set.indices]

    summarize("Train", train_metas)
    summarize("Valid", val_metas)

    check_overlap(
        [m["path"] for m in train_metas],
        [m["path"] for m in val_metas],
        "Exact path overlap"
    )

    print("=" * 80)
    print("Next step: Check video-level leakage using path structure")
    print("=" * 80)


if __name__ == "__main__":
    main()