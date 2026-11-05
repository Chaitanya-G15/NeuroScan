"""Evaluation and metric computation for Medical Anomaly Detection.

Features:
- Sample-level ROC-AUC and Precision-Recall AUC calculation.
- Optimal threshold calibration via Youden's J statistic / F1-score optimization.
- Generates pixel-level difference heatmaps and segmentations for individual test slices.
- Visual report generator saving diagnostic comparisons.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader

from backend.data.dataset import get_dataloaders
from backend.model.utils import apply_gaussian_blur, compute_residual_map, generate_heatmap_overlay
from backend.model.vae import BetaVAE


class AnomalyEvaluator:
    """Evaluates anomaly detection performance of a trained Beta-VAE model."""

    def __init__(
        self,
        model: BetaVAE,
        device: torch.device,
        blur_kernel: int = 5,
        blur_sigma: float = 1.5,
    ):
        """Initializes the evaluator.

        Args:
            model (BetaVAE): Trained model.
            device (torch.device): Device ('cuda' or 'cpu').
            blur_kernel (int): Smoothing kernel size for residual map.
            blur_sigma (float): Smoothing standard deviation.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

    def evaluate_dataloader(self, dataloader: DataLoader) -> Dict[str, Union[float, np.ndarray, List[float]]]:
        """Runs evaluation over a mixed validation/test set of healthy (0) and anomalous (1) scans.

        Args:
            dataloader (DataLoader): DataLoader yielding (x, label, path).

        Returns:
            Dict containing:
                - 'roc_auc': Receiver Operating Characteristic AUC.
                - 'pr_auc': Precision-Recall AUC.
                - 'optimal_threshold': Best score cutoff determined via Youden's J.
                - 'scores': List of float anomaly scores.
                - 'labels': Ground truth binary labels.
                - 'fpr', 'tpr': ROC curve points.
        """
        all_scores: List[float] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for x_batch, labels, _ in dataloader:
                x_batch = x_batch.to(self.device)
                recon_batch, _, _ = self.model(x_batch)

                for i in range(x_batch.size(0)):
                    x_single = x_batch[i : i + 1]
                    recon_single = recon_batch[i : i + 1]

                    # 1. Compute pixel residual map
                    res_map = compute_residual_map(x_single, recon_single)

                    # 2. Smooth residual map
                    smoothed = apply_gaussian_blur(res_map, self.blur_kernel, self.blur_sigma)

                    # 3. Anomaly score: mean of top 5% error pixels
                    sorted_err = np.sort(smoothed.ravel())
                    top_k = max(1, int(len(sorted_err) * 0.05))
                    score = float(np.mean(sorted_err[-top_k:]))

                    all_scores.append(score)
                    all_labels.append(int(labels[i].item()))

        scores_arr = np.array(all_scores, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int32)

        # Calculate ROC and PR curves
        fpr, tpr, roc_thresh = roc_curve(labels_arr, scores_arr)
        roc_auc_val = float(roc_auc_score(labels_arr, scores_arr))

        precision, recall, _ = precision_recall_curve(labels_arr, scores_arr)
        pr_auc_val = float(auc(recall, precision))

        # Optimal threshold using Youden's J statistic (sensitivity + specificity - 1)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        optimal_threshold = float(roc_thresh[best_idx]) if best_idx < len(roc_thresh) else 0.5

        return {
            "roc_auc": roc_auc_val,
            "pr_auc": pr_auc_val,
            "optimal_threshold": optimal_threshold,
            "scores": all_scores,
            "labels": all_labels,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    def inspect_single_image(
        self,
        image_tensor: torch.Tensor,
        raw_display_img: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Union[np.ndarray, float, bool]]:
        """Performs full end-to-end diagnostic pipeline on a single medical image.

        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor [1, C, H, W].
            raw_display_img (np.ndarray): Original image array [H, W] or [H, W, 3] uint8.
            threshold (float): Anomaly score cutoff.

        Returns:
            Dict containing:
                - 'reconstruction': Reconstructed image uint8 [H, W].
                - 'residual_map': Normalized residual map float [H, W].
                - 'heatmap_overlay': Thermal overlay RGB uint8 [H, W, 3].
                - 'binary_mask': Segmentation mask uint8 [H, W].
                - 'anomaly_score': Float severity score in [0.0, 1.0].
                - 'is_anomalous': Boolean classification according to threshold.
        """
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            recon_tensor = self.model.reconstruct(image_tensor)

        # 1. Pixel residual difference map
        raw_res = compute_residual_map(image_tensor, recon_tensor)
        smoothed_res = apply_gaussian_blur(raw_res, self.blur_kernel, self.blur_sigma)

        # 2. Thermal overlay and segmentation mask
        overlay_rgb, mask, anomaly_score = generate_heatmap_overlay(
            original_img=raw_display_img,
            anomaly_map=smoothed_res,
            threshold=threshold,
        )

        # 3. Format reconstructed image as uint8 for UI display
        recon_np = recon_tensor.squeeze(0).squeeze(0).cpu().numpy()
        recon_uint8 = (recon_np * 255.0).clip(0, 255).astype(np.uint8)

        return {
            "reconstruction": recon_uint8,
            "residual_map": smoothed_res,
            "heatmap_overlay": overlay_rgb,
            "binary_mask": mask,
            "anomaly_score": anomaly_score,
            "is_anomalous": bool(anomaly_score >= threshold),
        }
