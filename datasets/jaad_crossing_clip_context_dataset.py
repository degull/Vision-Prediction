# C:\Users\IIPL02\Desktop\Vision Prediction\datasets\jaad_crossing_clip_context_dataset.py
import os
import re
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class JAADCrossingClipContextDataset(Dataset):
    """
    JAAD clip-level dataset for Stage2.

    Added:
      - cache for prebuilt clip metadata
      - optional cache refresh
      - much faster re-runs after first build

    Returns:
        {
            "video": Tensor [T, 3, H, W],
            "video_path": str,
            "video_id": str,
            "clip_start": int,
            "clip_end": int,
            "frame_indices": LongTensor [T],
            "crossing_label": Tensor scalar float,

            "attr_vec": Tensor [6],
            "app_vec": Tensor [5],
            "traffic_vec": Tensor [6],
            "vehicle_vec": Tensor [6],
        }
    """

    def __init__(
        self,
        clips_dir: str,
        annotations_dir: str = None,
        attributes_dir: str = None,
        appearance_dir: str = None,
        traffic_dir: str = None,
        vehicle_dir: str = None,
        num_frames: int = 8,
        image_size: int = 224,
        frame_stride: int = 1,
        sample_stride: int = 2,
        early_horizon: int = 30,
        verbose: bool = False,
        stride: int = None,   # backward compatibility
        use_cache: bool = True,
        rebuild_cache: bool = False,
        cache_dir: str = None,
        **kwargs,
    ):
        self.clips_dir = clips_dir
        self.annotations_dir = annotations_dir
        self.attributes_dir = attributes_dir
        self.appearance_dir = appearance_dir
        self.traffic_dir = traffic_dir
        self.vehicle_dir = vehicle_dir

        self.num_frames = num_frames
        self.frame_stride = frame_stride if stride is None else stride
        self.sample_stride = sample_stride
        self.early_horizon = early_horizon
        self.verbose = verbose

        self.use_cache = use_cache
        self.rebuild_cache = rebuild_cache
        self.cache_dir = cache_dir if cache_dir is not None else clips_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        self.video_files = sorted([
            os.path.join(clips_dir, f)
            for f in os.listdir(clips_dir)
            if f.lower().endswith(".mp4")
        ])

        if len(self.video_files) == 0:
            raise RuntimeError(f"No .mp4 files found in clips_dir: {clips_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.cache_path = self._build_cache_path()
        self.samples = []
        self.skipped_videos = []

        if self.use_cache and (not self.rebuild_cache) and os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._build_samples_from_scratch()
            if self.use_cache:
                self._save_cache()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No labeled clip samples found from JAAD annotations.\n"
                "Please inspect XML files and clip videos."
            )

        if verbose:
            pos = sum(int(s["crossing_label"] == 1.0) for s in self.samples)
            neg = sum(int(s["crossing_label"] == 0.0) for s in self.samples)
            unique_videos = len(set(s["video_id"] for s in self.samples))

            print("[JAADCrossingClipContextDataset]")
            print(f"  clips_dir       : {self.clips_dir}")
            print(f"  annotations_dir : {self.annotations_dir}")
            print(f"  attributes_dir  : {self.attributes_dir}")
            print(f"  appearance_dir  : {self.appearance_dir}")
            print(f"  traffic_dir     : {self.traffic_dir}")
            print(f"  vehicle_dir     : {self.vehicle_dir}")
            print(f"  num_frames      : {self.num_frames}")
            print(f"  frame_stride    : {self.frame_stride}")
            print(f"  sample_stride   : {self.sample_stride}")
            print(f"  total videos    : {unique_videos}")
            print(f"  total samples   : {len(self.samples)}")
            print(f"  positive        : {pos}")
            print(f"  negative        : {neg}")
            print(f"  skipped videos  : {len(self.skipped_videos)}")
            print(f"  cache_path      : {self.cache_path}")
            print(f"  cache_used      : {self.use_cache and os.path.exists(self.cache_path)}")

    def __len__(self):
        return len(self.samples)

    # ============================================================
    # Cache
    # ============================================================
    def _build_cache_path(self):
        clips_name = os.path.basename(os.path.normpath(self.clips_dir))
        cache_name = (
            f"jaad_clip_context_cache_"
            f"{clips_name}_"
            f"nf{self.num_frames}_"
            f"fs{self.frame_stride}_"
            f"ss{self.sample_stride}.pt"
        )
        return os.path.join(self.cache_dir, cache_name)

    def _save_cache(self):
        cache_obj = {
            "samples": self.samples,
            "skipped_videos": self.skipped_videos,
            "meta": {
                "clips_dir": self.clips_dir,
                "annotations_dir": self.annotations_dir,
                "attributes_dir": self.attributes_dir,
                "appearance_dir": self.appearance_dir,
                "traffic_dir": self.traffic_dir,
                "vehicle_dir": self.vehicle_dir,
                "num_frames": self.num_frames,
                "frame_stride": self.frame_stride,
                "sample_stride": self.sample_stride,
                "early_horizon": self.early_horizon,
            }
        }
        torch.save(cache_obj, self.cache_path)
        if self.verbose:
            print(f"[Cache Saved] {self.cache_path}")

    def _load_cache(self):
        cache_obj = torch.load(self.cache_path, map_location="cpu")
        self.samples = cache_obj["samples"]
        self.skipped_videos = cache_obj.get("skipped_videos", [])
        if self.verbose:
            print(f"[Cache Loaded] {self.cache_path}")
            print(f"  cached samples  : {len(self.samples)}")
            print(f"  skipped videos  : {len(self.skipped_videos)}")

    def _build_samples_from_scratch(self):
        if self.verbose:
            print("[Cache Miss] Building dataset from scratch...")

        skipped = []
        samples = []

        for video_path in self.video_files:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            crossing_label = self._load_crossing_label(video_id)

            if crossing_label is None:
                skipped.append(video_path)
                continue

            num_total_frames = self._get_num_frames(video_path)
            if num_total_frames <= 0:
                skipped.append(video_path)
                continue

            attr_vec = self._load_attr_vector(video_id)
            app_vec = self._load_app_vector(video_id)
            traffic_vec = self._load_traffic_vector(video_id)
            vehicle_vec = self._load_vehicle_vector(video_id)

            frame_indices_list = self._build_clip_frame_indices(num_total_frames)

            if len(frame_indices_list) == 0:
                skipped.append(video_path)
                continue

            for frame_indices in frame_indices_list:
                clip_start = int(frame_indices[0].item())
                clip_end = int(frame_indices[-1].item())

                samples.append({
                    "video_path": video_path,
                    "video_id": video_id,
                    "crossing_label": float(crossing_label),
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                    "frame_indices": frame_indices.clone().long(),
                    "attr_vec": attr_vec.clone().float(),
                    "app_vec": app_vec.clone().float(),
                    "traffic_vec": traffic_vec.clone().float(),
                    "vehicle_vec": vehicle_vec.clone().float(),
                })

        self.samples = samples
        self.skipped_videos = skipped

    # ============================================================
    # Clip construction
    # ============================================================
    def _get_num_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(frame_count, 0)

    def _build_clip_frame_indices(self, num_total_frames):
        needed = 1 + (self.num_frames - 1) * self.frame_stride
        frame_indices_list = []

        if num_total_frames <= 0:
            return frame_indices_list

        if num_total_frames < needed:
            indices = list(range(0, num_total_frames, self.frame_stride))
            if len(indices) == 0:
                indices = [0]
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            indices = indices[:self.num_frames]
            frame_indices_list.append(torch.tensor(indices, dtype=torch.long))
            return frame_indices_list

        max_start = num_total_frames - needed
        start_positions = list(range(0, max_start + 1, self.sample_stride))

        if len(start_positions) == 0 or start_positions[-1] != max_start:
            start_positions.append(max_start)

        for start in start_positions:
            indices = [start + i * self.frame_stride for i in range(self.num_frames)]
            frame_indices_list.append(torch.tensor(indices, dtype=torch.long))

        return frame_indices_list

    # ============================================================
    # Video loading
    # ============================================================
    def _read_clip_frames(self, video_path, frame_indices):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        images = []
        last_valid = None

        for idx in frame_indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                last_valid = img
                images.append(img)
            else:
                if last_valid is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret0, frame0 = cap.read()
                    if not ret0:
                        cap.release()
                        raise ValueError(f"Video has no readable frames: {video_path}")
                    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                    last_valid = Image.fromarray(frame0)
                images.append(last_valid)

        cap.release()

        while len(images) < self.num_frames:
            images.append(images[-1])

        if len(images) > self.num_frames:
            images = images[:self.num_frames]

        return images

    # ============================================================
    # XML path helpers
    # ============================================================
    def _candidate_xml_paths(self, video_id, folder, suffixes=None):
        if folder is None:
            return []

        if suffixes is None:
            suffixes = [
                f"{video_id}.xml",
                f"{video_id}_annotation.xml",
                f"{video_id}_annotations.xml",
                f"{video_id}_attributes.xml",
                f"{video_id}_appearance.xml",
                f"{video_id}_traffic.xml",
                f"{video_id}_vehicle.xml",
            ]

        paths = [os.path.join(folder, s) for s in suffixes]
        return [p for p in paths if os.path.exists(p)]

    def _load_xml_root(self, xml_paths):
        for xml_path in xml_paths:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                return root, xml_path
            except Exception as e:
                if self.verbose:
                    print(f"[XML Parse Failed] {xml_path} | {repr(e)}")
                continue
        return None, None

    # ============================================================
    # Stage1 label extraction
    # ============================================================
    def _safe_int01(self, value):
        if value is None:
            return None

        if isinstance(value, (int, float)):
            v = int(value)
            if v in [0, 1]:
                return v
            return None

        s = str(value).strip().lower()

        if s in ["0", "no", "false", "not_crossing", "noncrossing", "non-crossing"]:
            return 0
        if s in ["1", "yes", "true", "crossing", "cross"]:
            return 1

        return None

    def _extract_from_element(self, elem):
        candidate_keys = [
            "crossing",
            "cross",
            "action",
            "behavior",
            "intention",
            "decision",
            "crossing_label",
            "pedestrian_crossing",
        ]

        for key in candidate_keys:
            if key in elem.attrib:
                value = self._safe_int01(elem.attrib.get(key))
                if value is not None:
                    return value

        for child in elem:
            tag = child.tag.lower()
            text = child.text.strip() if child.text is not None else None

            for key in candidate_keys:
                if key in tag:
                    value = self._safe_int01(text)
                    if value is not None:
                        return value

        return None

    def _load_crossing_label(self, video_id):
        xml_paths = self._candidate_xml_paths(
            video_id=video_id,
            folder=self.annotations_dir,
            suffixes=[
                f"{video_id}.xml",
                f"{video_id}_annotation.xml",
                f"{video_id}_annotations.xml",
            ]
        )

        xml_paths += self._candidate_xml_paths(
            video_id=video_id,
            folder=self.attributes_dir,
            suffixes=[
                f"{video_id}.xml",
                f"{video_id}_attributes.xml",
            ]
        )

        if self.verbose:
            print(f"[Label XML Candidates] {video_id} -> {xml_paths}")

        for xml_path in xml_paths:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                if self.verbose:
                    print(f"[XML Parse Failed] {xml_path} | {repr(e)}")
                continue

            value = self._extract_from_element(root)
            if value is not None:
                if self.verbose:
                    print(f"[Label Found] {video_id} | {xml_path} | root -> {value}")
                return value

            for elem in root.iter():
                value = self._extract_from_element(elem)
                if value is not None:
                    if self.verbose:
                        print(f"[Label Found] {video_id} | {xml_path} | elem={elem.tag} -> {value}")
                    return value

        if self.verbose:
            print(
                f"[Crossing Label Not Found] video_id={video_id} | "
                f"xml={xml_paths[0] if len(xml_paths) > 0 else 'None'}"
            )

        return None

    # ============================================================
    # Generic XML flattening for context vectors
    # ============================================================
    def _normalize_text(self, x):
        if x is None:
            return None
        return str(x).strip().lower()

    def _safe_float(self, x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            s = str(x).strip().lower()
            m = re.search(r"[-+]?\d*\.?\d+", s)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    return None
        return None

    def _collect_xml_fields(self, root):
        fields = {}

        def add_value(key, value):
            key = self._normalize_text(key)
            value = self._normalize_text(value)
            if key is None or value is None or value == "":
                return
            fields.setdefault(key, []).append(value)

        for elem in root.iter():
            tag = self._normalize_text(elem.tag)

            if elem.text is not None:
                txt = self._normalize_text(elem.text)
                if txt not in [None, ""]:
                    add_value(tag, txt)

            for k, v in elem.attrib.items():
                add_value(k, v)
                if tag is not None:
                    add_value(f"{tag}.{k}", v)

        return fields

    def _has_positive_keyword(self, fields, keys, positive_terms=None):
        if positive_terms is None:
            positive_terms = ["1", "yes", "true", "on", "present", "crossing", "walking"]

        keys = [k.lower() for k in keys]
        positive_terms = [p.lower() for p in positive_terms]

        for fk, values in fields.items():
            if any(k in fk for k in keys):
                for v in values:
                    if any(p == v or p in v for p in positive_terms):
                        return 1.0
        return 0.0

    def _numeric_feature(self, fields, keys, default=0.0, clip_min=None, clip_max=None, normalize_by=None):
        keys = [k.lower() for k in keys]
        vals = []

        for fk, values in fields.items():
            if any(k in fk for k in keys):
                for v in values:
                    fv = self._safe_float(v)
                    if fv is not None:
                        vals.append(fv)

        if len(vals) == 0:
            return float(default)

        x = max(vals)

        if clip_min is not None:
            x = max(x, clip_min)
        if clip_max is not None:
            x = min(x, clip_max)
        if normalize_by is not None and normalize_by > 0:
            x = x / normalize_by

        return float(x)

    def _load_context_root(self, video_id, folder, suffixes):
        xml_paths = self._candidate_xml_paths(video_id=video_id, folder=folder, suffixes=suffixes)
        root, xml_path = self._load_xml_root(xml_paths)

        if self.verbose:
            print(f"[Context XML] {video_id} | folder={folder} | path={xml_path}")

        return root

    # ============================================================
    # Context vector parsers
    # ============================================================
    def _load_attr_vector(self, video_id):
        root = self._load_context_root(
            video_id,
            self.attributes_dir,
            [f"{video_id}.xml", f"{video_id}_attributes.xml"]
        )

        if root is None:
            return torch.zeros(6, dtype=torch.float32)

        fields = self._collect_xml_fields(root)

        vec = [
            self._has_positive_keyword(fields, ["cross", "intention", "intent", "crossing"]),
            self._has_positive_keyword(fields, ["walk", "walking", "motion"], ["1", "yes", "true", "walking", "walk"]),
            self._has_positive_keyword(fields, ["stand", "standing", "still"], ["1", "yes", "true", "standing", "stand"]),
            self._has_positive_keyword(fields, ["look", "looking", "gaze", "attention"]),
            self._has_positive_keyword(fields, ["move", "moving", "motion"]),
            self._has_positive_keyword(fields, ["curb", "road", "boundary", "sidewalk"]),
        ]

        return torch.tensor(vec, dtype=torch.float32)

    def _load_app_vector(self, video_id):
        root = self._load_context_root(
            video_id,
            self.appearance_dir,
            [f"{video_id}.xml", f"{video_id}_appearance.xml"]
        )

        if root is None:
            return torch.zeros(5, dtype=torch.float32)

        fields = self._collect_xml_fields(root)

        visible = self._numeric_feature(
            fields, ["visible", "visibility", "ratio"],
            default=0.0, clip_min=0.0, clip_max=1.0
        )

        if visible > 1.0:
            visible = min(visible / 100.0, 1.0)

        vec = [
            self._has_positive_keyword(fields, ["front", "facing", "orientation"], ["front", "road", "forward"]),
            self._has_positive_keyword(fields, ["left", "orientation"], ["left"]),
            self._has_positive_keyword(fields, ["right", "orientation"], ["right"]),
            visible,
            self._has_positive_keyword(fields, ["occl", "occlusion", "occluded"]),
        ]

        return torch.tensor(vec, dtype=torch.float32)

    def _load_traffic_vector(self, video_id):
        root = self._load_context_root(
            video_id,
            self.traffic_dir,
            [f"{video_id}.xml", f"{video_id}_traffic.xml"]
        )

        if root is None:
            return torch.zeros(6, dtype=torch.float32)

        fields = self._collect_xml_fields(root)

        vec = [
            self._has_positive_keyword(fields, ["crosswalk", "zebra"]),
            self._has_positive_keyword(fields, ["signal", "light"], ["red"]),
            self._has_positive_keyword(fields, ["signal", "light"], ["green"]),
            self._has_positive_keyword(fields, ["signal", "light"], ["none", "unknown", "absent", "off"]),
            self._has_positive_keyword(fields, ["dense", "traffic", "busy", "congestion"]),
            self._has_positive_keyword(fields, ["intersection", "junction"]),
        ]

        return torch.tensor(vec, dtype=torch.float32)

    def _load_vehicle_vector(self, video_id):
        root = self._load_context_root(
            video_id,
            self.vehicle_dir,
            [f"{video_id}.xml", f"{video_id}_vehicle.xml"]
        )

        if root is None:
            return torch.zeros(6, dtype=torch.float32)

        fields = self._collect_xml_fields(root)

        vehicle_count = self._numeric_feature(
            fields, ["count", "vehicle_count", "num_vehicle", "cars"],
            default=0.0, clip_min=0.0, clip_max=20.0, normalize_by=10.0
        )
        vehicle_count = min(vehicle_count, 1.0)

        closest_dist = self._numeric_feature(
            fields, ["distance", "dist", "closest"],
            default=0.0, clip_min=0.0, clip_max=100.0, normalize_by=100.0
        )

        vec = [
            vehicle_count,
            closest_dist,
            self._has_positive_keyword(fields, ["approach", "approaching", "coming"]),
            self._has_positive_keyword(fields, ["left"], ["left"]),
            self._has_positive_keyword(fields, ["right"], ["right"]),
            self._has_positive_keyword(fields, ["front", "ahead"], ["front", "ahead"]),
        ]

        return torch.tensor(vec, dtype=torch.float32)

    # ============================================================
    # Get item
    # ============================================================
    def __getitem__(self, idx):
        sample = self.samples[idx]

        video_path = sample["video_path"]
        video_id = sample["video_id"]
        label = sample["crossing_label"]
        clip_start = sample["clip_start"]
        clip_end = sample["clip_end"]
        frame_indices = sample["frame_indices"]

        frames = self._read_clip_frames(video_path, frame_indices)
        if len(frames) == 0:
            raise ValueError(f"Video clip has no readable frames: {video_path}")

        video = torch.stack([self.transform(img) for img in frames], dim=0)

        return {
            "video": video,
            "video_path": video_path,
            "video_id": video_id,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "frame_indices": frame_indices.clone().long(),
            "crossing_label": torch.tensor(label, dtype=torch.float32),
            "attr_vec": sample["attr_vec"].clone().float(),
            "app_vec": sample["app_vec"].clone().float(),
            "traffic_vec": sample["traffic_vec"].clone().float(),
            "vehicle_vec": sample["vehicle_vec"].clone().float(),
        }


if __name__ == "__main__":
    dataset = JAADCrossingClipContextDataset(
        clips_dir=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD clips",
        annotations_dir=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations",
        attributes_dir=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations_attributes",
        appearance_dir=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations_appearance",
        traffic_dir=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations_traffic",
        vehicle_dir=r"C:\Users\IIPL02\Desktop\Vision Prediction\data\JAAD\JAAD annotations\annotations_vehicle",
        num_frames=8,
        image_size=224,
        frame_stride=1,
        sample_stride=2,
        verbose=True,
        use_cache=True,
        rebuild_cache=False,
    )

    sample = dataset[0]
    print("\n[Smoke Test]")
    print("video           :", tuple(sample["video"].shape))
    print("video_id        :", sample["video_id"])
    print("clip_start      :", sample["clip_start"])
    print("clip_end        :", sample["clip_end"])
    print("frame_indices   :", sample["frame_indices"].tolist())
    print("crossing_label  :", float(sample["crossing_label"].item()))
    print("attr_vec        :", tuple(sample["attr_vec"].shape))
    print("app_vec         :", tuple(sample["app_vec"].shape))
    print("traffic_vec     :", tuple(sample["traffic_vec"].shape))
    print("vehicle_vec     :", tuple(sample["vehicle_vec"].shape))