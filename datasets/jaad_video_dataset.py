import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class JAADVideoDataset(Dataset):
    def __init__(
        self,
        clips_dir: str,
        num_frames: int = 8,
        image_size: int = 224,
        stride: int = 1,
    ):
        self.clips_dir = clips_dir
        self.num_frames = num_frames
        self.stride = stride

        self.video_files = sorted([
            os.path.join(clips_dir, f)
            for f in os.listdir(clips_dir)
            if f.endswith(".mp4")
        ])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.video_files)

    def _read_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = self._read_video_frames(video_path)

        if len(frames) < self.num_frames:
            raise ValueError(f"Video too short: {video_path}, got {len(frames)} frames")

        # 일단 MVP에서는 앞의 8프레임만 사용
        selected = frames[:self.num_frames]

        selected = [self.transform(img) for img in selected]
        video = torch.stack(selected, dim=0)   # [T, 3, H, W]

        return {
            "video": video,
            "video_path": video_path,
        }