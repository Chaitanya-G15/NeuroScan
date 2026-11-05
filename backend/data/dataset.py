"""Medical Image Dataset and DataLoader pipeline for Brain MRI Anomaly Detection.

Implements:
- BrainMRIDataset: Loads grayscale/RGB brain MRI slices, splits healthy (normal) vs anomalous
  (tumor/lesion), applies standardized contrast-adjusted preprocessing without MONAI.
- SyntheticBrainMRIGenerator: Helper to auto-generate realistic synthetic healthy & tumor MRI slices
  for instant bootstrapping, testing, and training if an external raw dataset is downloading.
- get_dataloaders: Configures training loader (strictly healthy scans) and validation loader
  (both healthy and anomalous slices with binary ground truth labels).
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BrainMRIDataset(Dataset):
    """PyTorch Dataset for Brain MRI Anomaly Detection.

    Designed for unsupervised anomaly detection:
    - Train split contains strictly healthy (label=0) slices.
    - Test/Validation split contains both healthy (0) and anomalous (1) slices.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: List[int],
        img_size: int = 128,
        in_channels: int = 1,
        is_training: bool = False,
        transform: Optional[Callable] = None,
    ):
        """Initializes the BrainMRIDataset.

        Args:
            image_paths (List[Union[str, Path]]): File paths to MRI images.
            labels (List[int]): Ground truth binary labels (0 = healthy, 1 = anomalous/tumor).
            img_size (int): Target spatial dimension (H=W). Default: 128.
            in_channels (int): 1 for single-channel grayscale, 3 for RGB. Default: 1.
            is_training (bool): If True, applies data augmentations (flips, slight rotations).
            transform (Optional[Callable]): Optional custom transform pipeline.
        """
        assert len(image_paths) == len(labels), "image_paths and labels must have the same length"
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.img_size = img_size
        self.in_channels = in_channels
        self.is_training = is_training

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_default_transforms()

    def _build_default_transforms(self) -> transforms.Compose:
        """Constructs torchvision transform pipeline based on training vs eval split."""
        transform_list = []
        if self.is_training:
            # Subtle augmentations preserving anatomical plausibility (avoid extreme warps)
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Loads, preprocesses, and returns a single MRI sample.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, int, str]:
                - tensor_img: Preprocessed image tensor [C, H, W] in [0.0, 1.0].
                - label: 0 for healthy, 1 for anomalous.
                - path: Path to original file string.
        """
        path = str(self.image_paths[idx])
        label = self.labels[idx]

        # Read image using OpenCV (strictly no MONAI)
        read_flag = cv2.IMREAD_GRAYSCALE if self.in_channels == 1 else cv2.IMREAD_COLOR
        img = cv2.imread(path, read_flag)

        if img is None:
            raise FileNotFoundError(f"Failed to load image at: {path}")

        # Resize to target dimension
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # Min-Max Normalization to [0.0, 1.0]
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        # Convert numpy array [H, W] or [H, W, C] to PyTorch tensor [C, H, W]
        if self.in_channels == 1:
            tensor_img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        else:
            # OpenCV BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        # Apply augmentation if any
        tensor_img = self.transform(tensor_img)

        return tensor_img, label, path


class SyntheticBrainMRIGenerator:
    """Generates realistic synthetic brain MRI slices for zero-dependency bootstrapping & testing.

    Creates realistic synthetic phantom heads (elliptical skull, brain parenchyma, ventricles)
    with optional hyperintense or hypointense tumor lesions with fuzzy infiltrative borders.
    """

    @staticmethod
    def generate_slice(img_size: int = 128, has_anomaly: bool = False, seed: Optional[int] = None) -> np.ndarray:
        """Generates a synthetic axial brain MRI slice [H, W] in range [0, 255] (uint8).

        Args:
            img_size (int): Dimension of image square.
            has_anomaly (bool): If True, introduces an irregular tumor/lesion anomaly.
            seed (Optional[int]): Random seed for reproducible generation.

        Returns:
            np.ndarray: Grayscale synthetic MRI slice [img_size, img_size] as uint8.
        """
        if seed is not None:
            np.random.seed(seed)

        canvas = np.zeros((img_size, img_size), dtype=np.float32)
        center_x, center_y = img_size // 2, img_size // 2

        # 1. Outer Skull (Bright bone signal in CT/T1)
        skull_rx = int(img_size * 0.42)
        skull_ry = int(img_size * 0.46)
        cv2.ellipse(canvas, (center_x, center_y), (skull_rx, skull_ry), 0, 0, 360, 0.4, -1)

        # 2. Brain Parenchyma (Cortex & White Matter)
        brain_rx = int(img_size * 0.38)
        brain_ry = int(img_size * 0.42)
        cv2.ellipse(canvas, (center_x, center_y), (brain_rx, brain_ry), 0, 0, 360, 0.65, -1)

        # 3. Add subtle brain sulci texture / Perlin-like structural noise
        noise = np.random.normal(0, 0.03, (img_size, img_size)).astype(np.float32)
        canvas = np.clip(canvas + noise * (canvas > 0.1), 0.0, 1.0)

        # 4. Ventricles (CSF - Cerebrospinal Fluid, dark in T1-weighted)
        v_offset = int(img_size * 0.06)
        v_h = int(img_size * 0.14)
        v_w = int(img_size * 0.04)
        cv2.ellipse(canvas, (center_x - v_offset, center_y), (v_w, v_h), -10, 0, 360, 0.15, -1)
        cv2.ellipse(canvas, (center_x + v_offset, center_y), (v_w, v_h), 10, 0, 360, 0.15, -1)

        # 5. Anomalous Tumor Lesion (if specified)
        if has_anomaly:
            # Choose random lesion center within brain hemisphere
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.12, 0.26) * img_size
            t_x = int(center_x + radius * np.cos(angle))
            t_y = int(center_y + radius * np.sin(angle))
            t_radius = int(np.random.uniform(0.06, 0.12) * img_size)

            # High intensity irregular tumor core with edema halo
            tumor_mask = np.zeros((img_size, img_size), dtype=np.float32)
            cv2.circle(tumor_mask, (t_x, t_y), t_radius, 0.95, -1)
            # Add irregular protrusion
            p_x = int(t_x + np.random.uniform(-t_radius * 0.5, t_radius * 0.5))
            p_y = int(t_y + np.random.uniform(-t_radius * 0.5, t_radius * 0.5))
            cv2.circle(tumor_mask, (p_x, p_y), int(t_radius * 0.7), 0.85, -1)

            # Blur lesion border to simulate infiltrative tissue / edema
            tumor_mask = cv2.GaussianBlur(tumor_mask, (11, 11), 3.0)
            # Blend into parenchyma
            canvas = np.maximum(canvas, tumor_mask)

        # Smooth slightly and convert to uint8 [0, 255]
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)
        canvas_uint8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
        return canvas_uint8

    @classmethod
    def generate_synthetic_dataset(
        cls,
        output_dir: Union[str, Path],
        num_healthy: int = 150,
        num_anomalous: int = 50,
        img_size: int = 128,
    ) -> Tuple[List[str], List[str]]:
        """Generates and writes synthetic healthy and anomalous datasets to disk.

        Args:
            output_dir (Union[str, Path]): Base directory to store dataset.
            num_healthy (int): Number of normal healthy scans to generate.
            num_anomalous (int): Number of tumorous/anomalous scans to generate.
            img_size (int): Spatial dimensions.

        Returns:
            Tuple[List[str], List[str]]: (List of healthy file paths, List of anomalous file paths)
        """
        base_dir = Path(output_dir)
        healthy_dir = base_dir / "healthy"
        anomaly_dir = base_dir / "anomalous"

        healthy_dir.mkdir(parents=True, exist_ok=True)
        anomaly_dir.mkdir(parents=True, exist_ok=True)

        healthy_paths = []
        for i in range(num_healthy):
            fpath = healthy_dir / f"healthy_mri_{i:04d}.png"
            if not fpath.exists():
                img = cls.generate_slice(img_size=img_size, has_anomaly=False, seed=i)
                cv2.imwrite(str(fpath), img)
            healthy_paths.append(str(fpath))

        anomaly_paths = []
        for i in range(num_anomalous):
            fpath = anomaly_dir / f"anomalous_mri_{i:04d}.png"
            if not fpath.exists():
                img = cls.generate_slice(img_size=img_size, has_anomaly=True, seed=1000 + i)
                cv2.imwrite(str(fpath), img)
            anomaly_paths.append(str(fpath))

        return healthy_paths, anomaly_paths


def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    img_size: int = 128,
    in_channels: int = 1,
    train_val_split: float = 0.8,
    num_workers: int = 0,
    auto_generate_synthetic: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares training and validation DataLoaders adhering to unsupervised protocol.

    Protocol:
    - Train Loader: Exclusively healthy scans (label 0) to learn normal manifold P(X_healthy).
    - Val Loader: Mixture of remaining healthy scans and anomalous/tumor scans (label 1).

    Args:
        data_dir (Union[str, Path]): Root directory of MRI scans.
        batch_size (int): Batch size.
        img_size (int): Image spatial dimensions.
        in_channels (int): 1 for grayscale, 3 for RGB.
        train_val_split (float): Ratio of healthy scans to allocate to training.
        num_workers (int): PyTorch DataLoader worker count.
        auto_generate_synthetic (bool): If True and directory empty, generate synthetic brain scans.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    root = Path(data_dir)
    healthy_dir = root / "healthy"
    anomaly_dir = root / "anomalous"

    # Auto-generate synthetic samples if folders don't exist or are empty
    valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    healthy_files = [str(p) for p in healthy_dir.glob("*") if p.suffix.lower() in valid_extensions] if healthy_dir.exists() else []
    anomaly_files = [str(p) for p in anomaly_dir.glob("*") if p.suffix.lower() in valid_extensions] if anomaly_dir.exists() else []

    if (len(healthy_files) == 0 or len(anomaly_files) == 0) and auto_generate_synthetic:
        healthy_files, anomaly_files = SyntheticBrainMRIGenerator.generate_synthetic_dataset(
            output_dir=root,
            num_healthy=160,
            num_anomalous=40,
            img_size=img_size,
        )

    # Split healthy data into train and validation sets
    np.random.seed(42)
    shuffled_healthy = np.random.permutation(healthy_files)
    split_idx = int(len(shuffled_healthy) * train_val_split)

    train_healthy = list(shuffled_healthy[:split_idx])
    val_healthy = list(shuffled_healthy[split_idx:])

    # 1. Training set: Strictly Healthy
    train_paths = train_healthy
    train_labels = [0] * len(train_paths)

    train_dataset = BrainMRIDataset(
        image_paths=train_paths,
        labels=train_labels,
        img_size=img_size,
        in_channels=in_channels,
        is_training=True,
    )

    # 2. Validation set: Both Healthy and Anomalous (for evaluating anomaly detection ROC-AUC)
    val_paths = val_healthy + anomaly_files
    val_labels = ([0] * len(val_healthy)) + ([1] * len(anomaly_files))

    val_dataset = BrainMRIDataset(
        image_paths=val_paths,
        labels=val_labels,
        img_size=img_size,
        in_channels=in_channels,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
