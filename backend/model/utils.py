"""Utility functions for VAE latent space exploration, image preprocessing, and anomaly mapping.

Contains:
- preprocess_image: Ingestion and normalization of medical image inputs (NumPy/Bytes to PyTorch).
- compute_residual_map: Pixel-wise reconstruction error calculation.
- apply_gaussian_blur: Spatial smoothing for residual maps to suppress high-frequency noise.
- generate_heatmap_overlay: Thermal jet overlay rendering with alpha blending on original scans.
- sample_latent_space: Interpolation and sampling across VAE latent space for visualization.
"""

from typing import Tuple, Union
import base64
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def preprocess_image(
    image_input: Union[bytes, str, np.ndarray],
    img_size: int = 128,
    in_channels: int = 1,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Ingests raw image data and converts to normalized PyTorch tensor and display array.

    Args:
        image_input (Union[bytes, str, np.ndarray]): Raw file bytes, file path, or numpy array.
        img_size (int): Target square dimension (H=W). Default: 128.
        in_channels (int): 1 for grayscale, 3 for RGB. Default: 1.

    Returns:
        Tuple[torch.Tensor, np.ndarray]:
            - tensor_img: Normalized PyTorch tensor of shape [1, C, H, W] in range [0.0, 1.0].
            - display_img: uint8 numpy image of shape [H, W] or [H, W, 3] in range [0, 255].
    """
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        read_flag = cv2.IMREAD_GRAYSCALE if in_channels == 1 else cv2.IMREAD_COLOR
        raw_img = cv2.imdecode(nparr, read_flag)
    elif isinstance(image_input, str):
        read_flag = cv2.IMREAD_GRAYSCALE if in_channels == 1 else cv2.IMREAD_COLOR
        raw_img = cv2.imread(image_input, read_flag)
    elif isinstance(image_input, np.ndarray):
        raw_img = image_input.copy()
        if in_channels == 1 and len(raw_img.shape) == 3:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image_input type: {type(image_input)}")

    if raw_img is None:
        raise ValueError("Could not decode or read image.")

    # Resize to model resolution
    resized = cv2.resize(raw_img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Min-Max Normalization to [0.0, 1.0]
    img_float = resized.astype(np.float32)
    min_v, max_v = img_float.min(), img_float.max()
    if max_v > min_v:
        normalized = (img_float - min_v) / (max_v - min_v)
    else:
        normalized = np.zeros_like(img_float, dtype=np.float32)

    # Tensor formatting: [1, C, H, W]
    if in_channels == 1:
        tensor_img = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        display_img = (normalized * 255.0).astype(np.uint8)
    else:
        if len(normalized.shape) == 2:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        tensor_img = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        display_img = (normalized * 255.0).astype(np.uint8)

    return tensor_img, display_img


def compute_residual_map(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    metric: str = "abs_diff",
) -> np.ndarray:
    """Calculates pixel-level reconstruction error between original and reconstructed tensors.

    Uses absolute or squared error in natural [0.0, 1.0] intensity difference space.
    Does NOT divide by local max, preserving true absolute magnitude so healthy scans
    maintain low residuals instead of having minor noise amplified to 1.0.

    Args:
        original (torch.Tensor): Original image tensor [1, C, H, W] in [0, 1].
        reconstructed (torch.Tensor): Reconstructed tensor [1, C, H, W] in [0, 1].
        metric (str): 'abs_diff' (L1) or 'squared_diff' (MSE).

    Returns:
        np.ndarray: Absolute residual map array of shape [H, W] in natural [0.0, 1.0] scale.
    """
    diff = original - reconstructed  # [1, C, H, W]
    if metric == "squared_diff":
        error = torch.mean(diff ** 2, dim=1).squeeze(0)
    else:
        # L1 absolute difference represents actual pixel error in [0, 1]
        error = torch.mean(torch.abs(diff), dim=1).squeeze(0)

    error_np = error.detach().cpu().numpy()
    return np.clip(error_np, 0.0, 1.0)


def apply_gaussian_blur(error_map: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Applies Gaussian smoothing to suppress high-frequency acquisition noise while preserving true error scale.

    Args:
        error_map (np.ndarray): 2D float error map [H, W] in [0, 1].
        kernel_size (int): Size of Gaussian kernel (must be odd).
        sigma (float): Gaussian blur standard deviation.

    Returns:
        np.ndarray: Smoothed error map [H, W] in [0.0, 1.0].
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(error_map, (kernel_size, kernel_size), sigma)
    return np.clip(smoothed, 0.0, 1.0)


def generate_heatmap_overlay(
    original_img: np.ndarray,
    anomaly_map: np.ndarray,
    threshold: float = 0.25,
    alpha: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates thermal Jet heatmap overlay and binary segmentation mask for anomalous regions.

    Args:
        original_img (np.ndarray): Grayscale or RGB image [H, W] or [H, W, 3] uint8.
        anomaly_map (np.ndarray): Error map [H, W] float in [0.0, 1.0].
        threshold (float): Anomaly score cutoff [0.0, 1.0] above which pixels are marked positive.
        alpha (float): Overlay blend transparency factor.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - overlay_rgb: Color blended diagnostic image [H, W, 3] uint8.
            - binary_mask: Binary anomaly mask [H, W] uint8 (0 or 255).
            - anomaly_score: Summary metric representing anomaly severity (mean error of top 3% pixels).
    """
    H, W = anomaly_map.shape[:2]

    # Convert base image to 3-channel RGB uint8 for colormap compositing
    if len(original_img.shape) == 2:
        base_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = original_img.copy()

    if base_rgb.shape[:2] != (H, W):
        base_rgb = cv2.resize(base_rgb, (W, H), interpolation=cv2.INTER_AREA)

    # Scale anomaly map to [0, 255]
    heatmap_uint8 = (np.clip(anomaly_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Binary segmentation mask strictly where error exceeds threshold
    mask_binary = (anomaly_map >= threshold).astype(np.uint8) * 255

    # Alpha blend: apply intense thermal overlay only to pixels exceeding threshold
    mask_3d = np.repeat((anomaly_map >= threshold)[:, :, np.newaxis], 3, axis=2)
    overlay_rgb = base_rgb.copy()
    blended_region = (alpha * heatmap_rgb + (1 - alpha) * base_rgb).astype(np.uint8)
    overlay_rgb[mask_3d] = blended_region[mask_3d]

    # Summary Anomaly Score: Mean intensity of top 3% most anomalous pixels
    sorted_pixels = np.sort(anomaly_map.ravel())
    top_k = int(max(1, len(sorted_pixels) * 0.03))
    anomaly_score = float(np.mean(sorted_pixels[-top_k:]))

    return overlay_rgb, mask_binary, anomaly_score


def image_to_base64(img: np.ndarray, format: str = "PNG") -> str:
    """Converts a NumPy image [H, W] or [H, W, 3] to a Base64 data URL string for API responses.

    Args:
        img (np.ndarray): Image array.
        format (str): Image format ('PNG' or 'JPEG').

    Returns:
        str: Base64 data URI string.
    """
    if len(img.shape) == 2:
        pil_img = Image.fromarray(img, mode="L")
    else:
        pil_img = Image.fromarray(img, mode="RGB")

    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def sample_latent_space(
    model: torch.nn.Module,
    latent_dim: int,
    num_samples: int = 8,
    device: str = "cpu",
) -> torch.Tensor:
    """Draws random vectors from standard Gaussian prior z ~ N(0, I) and decodes them.

    Args:
        model (torch.nn.Module): Trained BetaVAE model with decoder.
        latent_dim (int): Dimensionality of latent space.
        num_samples (int): Number of synthetic scans to generate.
        device (str): Compute device.

    Returns:
        torch.Tensor: Decoded synthetic healthy scans [num_samples, C, H, W].
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        generated = model.decoder(z)
    return generated
