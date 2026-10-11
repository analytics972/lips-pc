"""
Author: Tomer Porat

Custom dataset loaders for real-world industrial datasets
Supports: RIAWELC (welding defect detection)
"""

import os
import re
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from config import RIAWELC_SPLIT_BY
except ImportError:
    RIAWELC_SPLIT_BY = "radiograph"


def _film_id(fname, level="radiograph"):
    """Group ID of a RIAWELC crop, at the requested granularity.

    Filenames look like 'RRT-101R_Img1_A80_S1_[12][10].png' and encode a
    3-level hierarchy:
        RRT-101R          = the physical WELD (only ~10-24 per class!)
        RRT-101R_Img1     = one X-ray exposure/radiograph of that weld
        [12][10]          = crop position inside the exposure
    Crops of the same exposure are near-duplicates; different exposures of
    the same weld are still strongly correlated (same weld geometry, same
    defects re-photographed). "weld" is therefore the strictest clean split.
    """
    if level == "weld":
        return fname.split('_')[0]
    m = re.match(r"(RRT-[^_]+_Img\d+)", fname)
    return m.group(1) if m else fname.split('_')[0]


class RIAWELCDataset(Dataset):
    """
    RIAWELC: X-ray images of welding defects
    - Source: https://github.com/stefyste/RIAWELC
    - 24,407 grayscale X-ray images (224×224)
    - 4 classes: ND (no defect), CR (crack), LP (lack of penetration), PO (porosity)
    - License: Freely released
    """

    CLASS_NAMES = ['ND', 'CR', 'LP', 'PO']  # no defect, crack, lack of penetration, porosity
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __init__(self, root_dir, train=True, transform=None, download=False):
        """
        Args:
            root_dir: Path to RIAWELC root directory (contains ND/, CR/, LP/, PO/ folders)
            train: If True, use 80% for training; if False, use 20% for testing
            transform: Optional image transformations
            download: Ignored (kept for compatibility with torchvision API)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # First pass: collect every crop of every class
        class_files = {}
        for class_name in self.CLASS_TO_IDX:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            class_files[class_name] = sorted(
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg')))

        if RIAWELC_SPLIT_BY in ("radiograph", "weld"):
            # Group split at the source-radiograph (film) level, GLOBAL across
            # classes: ALL crops of a film go to the same side of the split.
            #
            # Why: crops of the same film are near-duplicates (same exposure,
            # same weld, same film noise). The old crop-level random split put
            # 97.8% of test crops on films also seen in training, inflating
            # accuracy. A film may also contribute crops to SEVERAL classes,
            # so the split must be decided once per film, not per class,
            # or the same film still ends up on both sides.
            film_crops = {}   # film_id -> {class_name: n_crops}
            for class_name, files in class_files.items():
                for f in files:
                    fid = _film_id(f, RIAWELC_SPLIT_BY)
                    film_crops.setdefault(fid, {}).setdefault(class_name, 0)
                    film_crops[fid][class_name] += 1

            film_ids = sorted(film_crops)
            random.Random(42).shuffle(film_ids)

            target = {c: 0.2 * len(class_files[c]) for c in class_files}
            test_count = {c: 0 for c in class_files}
            test_films = set()
            for fid in film_ids:
                # add the film to test only if every class it touches still
                # needs test samples (keeps the split ~stratified at 20%)
                if all(test_count[c] < target[c] for c in film_crops[fid]):
                    test_films.add(fid)
                    for c, n in film_crops[fid].items():
                        test_count[c] += n

            for class_name, class_idx in self.CLASS_TO_IDX.items():
                for f in class_files[class_name]:
                    in_test = _film_id(f, RIAWELC_SPLIT_BY) in test_films
                    if in_test != train:
                        self.images.append(os.path.join(root_dir, class_name, f))
                        self.labels.append(class_idx)
        else:
            # Legacy crop-level split (kept for ROLLBACK / reproducing old
            # results). Set RIAWELC_SPLIT_BY = "crop" in config.py.
            for class_name, class_idx in self.CLASS_TO_IDX.items():
                train_files, test_files = train_test_split(
                    class_files[class_name],
                    test_size=0.2,
                    random_state=42,  # Fixed seed for reproducibility
                    shuffle=True       # Randomize within class
                )
                for f in (train_files if train else test_files):
                    self.images.append(os.path.join(root_dir, class_name, f))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image as grayscale
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label


class VISADataset(Dataset):
    """
    VISA: Video Anomaly Segmentation and Alignment
    - 10,821 industrial images across 12 categories and 3 domains
    - From Amazon Research (CVPR 2022)
    - License: CC BY 4.0

    Note: This is a placeholder. For full integration, download from:
    https://registry.opendata.aws/visa/
    """

    CLASS_NAMES = [
        'candle', 'cashew', 'chewinggum', 'fryingpan',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
        'pipe_fryable', 'pipe_non_fryable', 'pipe', 'wooden_spools'
    ]

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir: Path to VISA root directory
            train: If True, use training split; if False, use test split
            transform: Optional image transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        # Implementation would follow similar pattern to RIAWELC
        raise NotImplementedError("VISA dataset integration requires downloading from AWS. See comments for details.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raise NotImplementedError()
