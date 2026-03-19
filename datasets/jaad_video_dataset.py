import os
import re
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class JAADVideoDataset(Dataset):
    """
    JAAD video dataset with:
      - crossing label
      - structured context vectors

    Returns:
        {
            "video": Tensor [T, 3, H, W],
            "video_path": str,
            "video_id": str,
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
        sample_stride: int = 1,
        early_horizon: int = 30,
        verbose: bool = False,
        stride: int = None,   # backward compatibility
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

        # preload samples
        self.samples = []
        skipped = []

        for video_path in self.video_files:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            crossing_label = self._load_crossing_label(video_id)

            if crossing_label is None:
                skipped.append(video_path)
                continue

            attr_vec = self._load_attr_vector(video_id)
            app_vec = self._load_app_vector(video_id)
            traffic_vec = self._load_traffic_vector(video_id)
            vehicle_vec = self._load_vehicle_vector(video_id)

            self.samples.append({
                "video_path": video_path,
                "video_id": video_id,
                "crossing_label": float(crossing_label),
                "attr_vec": attr_vec,
                "app_vec": app_vec,
                "traffic_vec": traffic_vec,
                "vehicle_vec": vehicle_vec,
            })

        if len(self.samples) == 0:
            raise RuntimeError(
                "No labeled samples found from JAAD annotations.\n"
                "Please inspect XML files to confirm where the crossing label is stored."
            )

        if verbose:
            pos = sum(int(s["crossing_label"] == 1.0) for s in self.samples)
            neg = sum(int(s["crossing_label"] == 0.0) for s in self.samples)

            print("[JAADVideoDataset]")
            print(f"  clips_dir       : {self.clips_dir}")
            print(f"  annotations_dir : {self.annotations_dir}")
            print(f"  attributes_dir  : {self.attributes_dir}")
            print(f"  appearance_dir  : {self.appearance_dir}")
            print(f"  traffic_dir     : {self.traffic_dir}")
            print(f"  vehicle_dir     : {self.vehicle_dir}")
            print(f"  num_frames      : {self.num_frames}")
            print(f"  frame_stride    : {self.frame_stride}")
            print(f"  sample_stride   : {self.sample_stride}")
            print(f"  total samples   : {len(self.samples)}")
            print(f"  positive        : {pos}")
            print(f"  negative        : {neg}")
            if len(skipped) > 0:
                print(f"  skipped         : {len(skipped)}")

    def __len__(self):
        return len(self.samples)

    # ============================================================
    # Video loading
    # ============================================================
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

    def _sample_frames(self, frames):
        needed = 1 + (self.num_frames - 1) * self.frame_stride

        if len(frames) >= needed:
            selected = frames[:needed:self.frame_stride]
        else:
            selected = frames[::self.frame_stride]
            if len(selected) == 0:
                raise ValueError("No frames sampled from video.")
            while len(selected) < self.num_frames:
                selected.append(selected[-1])

        if len(selected) > self.num_frames:
            selected = selected[:self.num_frames]

        return selected

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
    # Label extraction
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

        # attributes
        for key in candidate_keys:
            if key in elem.attrib:
                value = self._safe_int01(elem.attrib.get(key))
                if value is not None:
                    return value

        # child tags
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

        # fallback to attributes if main annotation did not contain label
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
            print(f"[Label Not Found] {video_id}")

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
        """
        Flatten XML into key -> list[str] dictionary.
        This is intentionally generic because exact JAAD sub-annotation
        field names may vary.
        """
        fields = {}

        def add_value(key, value):
            key = self._normalize_text(key)
            value = self._normalize_text(value)
            if key is None or value is None or value == "":
                return
            fields.setdefault(key, []).append(value)

        for elem in root.iter():
            tag = self._normalize_text(elem.tag)

            # element text
            if elem.text is not None:
                txt = self._normalize_text(elem.text)
                if txt not in [None, ""]:
                    add_value(tag, txt)

            # attributes
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
        """
        attr_vec shape [6]
        Suggested semantics:
          0: crossing_intent
          1: walking
          2: standing
          3: looking
          4: moving
          5: near curb / road boundary
        """
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
        """
        app_vec shape [5]
        Suggested semantics:
          0: facing road / front
          1: left orientation
          2: right orientation
          3: visible ratio / visibility
          4: occlusion
        """
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
        """
        traffic_vec shape [6]
        Suggested semantics:
          0: crosswalk present
          1: signal red
          2: signal green
          3: signal absent/unknown
          4: dense traffic
          5: intersection/junction
        """
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
        """
        vehicle_vec shape [6]
        Suggested semantics:
          0: vehicle count normalized
          1: closest distance normalized (inverse-like proxy if needed)
          2: approaching vehicle
          3: vehicle on left
          4: vehicle on right
          5: vehicle in front
        """
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

        frames = self._read_video_frames(video_path)
        if len(frames) == 0:
            raise ValueError(f"Video has no readable frames: {video_path}")

        selected = self._sample_frames(frames)
        selected = [self.transform(img) for img in selected]
        video = torch.stack(selected, dim=0)   # [T, 3, H, W]

        return {
            "video": video,
            "video_path": video_path,
            "video_id": video_id,
            "crossing_label": torch.tensor(label, dtype=torch.float32),

            "attr_vec": sample["attr_vec"].clone().float(),
            "app_vec": sample["app_vec"].clone().float(),
            "traffic_vec": sample["traffic_vec"].clone().float(),
            "vehicle_vec": sample["vehicle_vec"].clone().float(),
        }