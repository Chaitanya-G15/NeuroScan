"""FastAPI REST Service for Medical Anomaly Detection.

Endpoints:
- GET  /api/v1/health       : Service and model status.
- POST /api/v1/detect       : Upload scan -> Reconstruct -> Residual Heatmap -> Anomaly Score.
- GET  /api/v1/sample       : Fetch pre-loaded healthy and tumor samples for instant demo testing.
- GET  /api/v1/latent-walk  : Explore latent space generation across dimensional interpolations.
- GET  /api/v1/metrics      : Dataset-level ROC-AUC and threshold validation performance.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.data.dataset import SyntheticBrainMRIGenerator, get_dataloaders
from backend.model.utils import (
    apply_gaussian_blur,
    compute_residual_map,
    generate_heatmap_overlay,
    image_to_base64,
    preprocess_image,
)
from backend.model.vae import BetaVAE
from backend.services.evaluate import AnomalyEvaluator

app = FastAPI(
    title="Medical Anomaly Detection API (Beta-VAE)",
    description="Unsupervised Rare Disease & Lesion Anomaly Detection System in Medical Imaging",
    version="1.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INSTANCE: Optional[BetaVAE] = None
EVALUATOR_INSTANCE: Optional[AnomalyEvaluator] = None
IMG_SIZE = 128
LATENT_DIM = 64
CHECKPOINT_PATH = Path("backend/checkpoints/best_vae_model.pt")


def load_or_init_model() -> BetaVAE:
    """Loads weights from checkpoint if available, or initializes model."""
    global MODEL_INSTANCE, EVALUATOR_INSTANCE
    model = BetaVAE(in_channels=1, latent_dim=LATENT_DIM, img_size=IMG_SIZE)

    if CHECKPOINT_PATH.exists():
        print(f"[*] Loading model weights from: {CHECKPOINT_PATH}")
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("[!] No pre-trained checkpoint found. Initialized with fresh weights.")

    model.to(DEVICE)
    model.eval()
    MODEL_INSTANCE = model
    EVALUATOR_INSTANCE = AnomalyEvaluator(model, DEVICE)
    return model


@app.on_event("startup")
def startup_event():
    """Initializes models and datasets on server startup."""
    load_or_init_model()


@app.get("/api/v1/health")
def health_check() -> Dict[str, Any]:
    """Returns system status, device information, and model loading state."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": MODEL_INSTANCE is not None,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "latent_dim": LATENT_DIM,
        "img_size": IMG_SIZE,
    }


@app.post("/api/v1/detect")
async def detect_anomaly(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    threshold: float = Form(0.28),
    blur_kernel: int = Form(5),
) -> Dict[str, Any]:
    """Performs inference: Reconstruction -> Residual Map -> Thermal Jet Overlay -> Diagnostic Report."""
    global MODEL_INSTANCE, EVALUATOR_INSTANCE
    if MODEL_INSTANCE is None:
        load_or_init_model()

    # 1. Acquire Image data (from uploaded file or pre-seeded sample)
    if file is not None:
        contents = await file.read()
        tensor_img, display_img = preprocess_image(contents, img_size=IMG_SIZE, in_channels=1)
    elif sample_id:
        # Generate on-the-fly sample
        has_anomaly = "tumor" in sample_id.lower() or "anomalous" in sample_id.lower()
        seed = int(sample_id.split("_")[-1]) if "_" in sample_id and sample_id.split("_")[-1].isdigit() else 42
        slice_img = SyntheticBrainMRIGenerator.generate_slice(img_size=IMG_SIZE, has_anomaly=has_anomaly, seed=seed)
        tensor_img, display_img = preprocess_image(slice_img, img_size=IMG_SIZE, in_channels=1)
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or sample_id.")

    # 2. VAE Reconstruction Pass
    tensor_img = tensor_img.to(DEVICE)
    with torch.no_grad():
        recon_tensor, mu, logvar = MODEL_INSTANCE(tensor_img)

    # 3. Residual Error & Anomaly Segmentation
    res_map = compute_residual_map(tensor_img, recon_tensor, metric="abs_diff")
    smoothed_map = apply_gaussian_blur(res_map, kernel_size=blur_kernel)
    overlay_rgb, binary_mask, anomaly_score = generate_heatmap_overlay(
        original_img=display_img,
        anomaly_map=smoothed_map,
        threshold=threshold,
    )

    recon_np = (recon_tensor.squeeze(0).squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # 4. Formulate Clinical Diagnostic Report
    is_anomaly = anomaly_score >= threshold
    severity = (
        "CRITICAL / HIGH RESIDUAL ANOMALY" if anomaly_score >= 0.40
        else "MODERATE LESION DEVIATION DETECTED" if anomaly_score >= threshold
        else "BENIGN / NORMAL ANATOMICAL PROFILE"
    )

    mu_list = mu.squeeze(0).cpu().tolist()

    return {
        "status": "success",
        "anomaly_score": round(anomaly_score, 4),
        "threshold": threshold,
        "is_anomalous": is_anomaly,
        "diagnostic_grade": severity,
        "latent_mu_preview": mu_list[:16],  # first 16 latent dimensions
        "images": {
            "original": image_to_base64(display_img),
            "reconstruction": image_to_base64(recon_np),
            "heatmap_overlay": image_to_base64(overlay_rgb),
            "binary_mask": image_to_base64(binary_mask),
        },
    }


@app.get("/api/v1/sample")
def get_demo_samples() -> Dict[str, Any]:
    """Provides sample IDs and previews for quick testing without manual uploads."""
    healthy_preview = SyntheticBrainMRIGenerator.generate_slice(img_size=IMG_SIZE, has_anomaly=False, seed=12)
    tumor_preview = SyntheticBrainMRIGenerator.generate_slice(img_size=IMG_SIZE, has_anomaly=True, seed=88)

    return {
        "samples": [
            {
                "id": "healthy_sample_12",
                "label": "Healthy Brain MRI (Normal Ventricles & Symmetry)",
                "type": "healthy",
                "preview": image_to_base64(healthy_preview),
            },
            {
                "id": "tumor_sample_88",
                "label": "Pathological Brain MRI (Hyperintense Lesion / Tumor)",
                "type": "anomalous",
                "preview": image_to_base64(tumor_preview),
            },
        ]
    }


@app.get("/api/v1/metrics")
def get_validation_metrics() -> Dict[str, Any]:
    """Computes and returns model validation metrics (ROC-AUC, PR-AUC)."""
    global MODEL_INSTANCE, EVALUATOR_INSTANCE
    if MODEL_INSTANCE is None:
        load_or_init_model()

    _, val_loader = get_dataloaders(
        data_dir="data/brain_mri",
        batch_size=16,
        img_size=IMG_SIZE,
        auto_generate_synthetic=True,
    )

    results = EVALUATOR_INSTANCE.evaluate_dataloader(val_loader)
    return {
        "roc_auc": round(results["roc_auc"], 4),
        "pr_auc": round(results["pr_auc"], 4),
        "optimal_threshold": round(results["optimal_threshold"], 4),
        "sample_count": len(results["labels"]),
        "fpr": [round(x, 4) for x in results["fpr"]],
        "tpr": [round(x, 4) for x in results["tpr"]],
    }
