"""Training loop for Beta-VAE with proper loss scaling and checkpoint management."""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from backend.data.dataset import get_dataloaders
from backend.model.vae import BetaVAE, vae_loss_function


class KLAnnealer:
    """Computes dynamic beta value across training epochs."""

    def __init__(
        self,
        target_beta: float = 1e-4,
        warmup_epochs: int = 5,
        start_beta: float = 0.0,
    ):
        self.target_beta = target_beta
        self.warmup_epochs = max(1, warmup_epochs)
        self.start_beta = start_beta

    def get_beta(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return self.target_beta
        fraction = epoch / self.warmup_epochs
        return self.start_beta + fraction * (self.target_beta - self.start_beta)


def train_epoch(
    model: BetaVAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float,
    loss_type: str = "mse",
) -> Dict[str, float]:
    model.train()
    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    kld_loss_accum = 0.0
    num_batches = 0

    for x_batch, _, _ in dataloader:
        x_batch = x_batch.to(device)
        optimizer.zero_grad()

        # Forward pass: Encode -> Reparameterize -> Decode
        recon_batch, mu, logvar = model(x_batch)

        # Compute combined VAE loss
        losses = vae_loss_function(
            recon_x=recon_batch,
            x=x_batch,
            mu=mu,
            logvar=logvar,
            beta=beta,
            loss_type=loss_type,
        )

        loss = losses["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss_accum += loss.item()
        recon_loss_accum += losses["recon_loss"].item()
        kld_loss_accum += losses["kld_loss"].item()
        num_batches += 1

    return {
        "loss": total_loss_accum / max(1, num_batches),
        "recon_loss": recon_loss_accum / max(1, num_batches),
        "kld_loss": kld_loss_accum / max(1, num_batches),
        "beta": beta,
    }


def validate_epoch(
    model: BetaVAE,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    loss_type: str = "mse",
) -> Dict[str, float]:
    model.eval()
    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    kld_loss_accum = 0.0
    healthy_batches = 0

    with torch.no_grad():
        for x_batch, labels, _ in dataloader:
            mask_healthy = labels == 0
            if not mask_healthy.any():
                continue

            x_healthy = x_batch[mask_healthy].to(device)
            recon_healthy, mu, logvar = model(x_healthy)

            losses = vae_loss_function(
                recon_x=recon_healthy,
                x=x_healthy,
                mu=mu,
                logvar=logvar,
                beta=beta,
                loss_type=loss_type,
            )

            total_loss_accum += losses["loss"].item()
            recon_loss_accum += losses["recon_loss"].item()
            kld_loss_accum += losses["kld_loss"].item()
            healthy_batches += 1

    return {
        "val_loss": total_loss_accum / max(1, healthy_batches),
        "val_recon_loss": recon_loss_accum / max(1, healthy_batches),
        "val_kld_loss": kld_loss_accum / max(1, healthy_batches),
    }


def train_vae(
    data_dir: str = "data/brain_mri",
    output_dir: str = "backend/checkpoints",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 2e-3,
    latent_dim: int = 64,
    img_size: int = 128,
    in_channels: int = 1,
    target_beta: float = 1e-4,
    warmup_epochs: int = 5,
    loss_type: str = "mse",
    device_name: Optional[str] = None,
) -> Dict[str, List[float]]:
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"[*] Initializing Beta-VAE training on device: {device}")

    # 1. Dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        in_channels=in_channels,
        auto_generate_synthetic=True,
    )
    print(f"[*] Loaded {len(train_loader.dataset)} training samples (healthy only).")
    print(f"[*] Loaded {len(val_loader.dataset)} validation samples (healthy + anomalous).")

    # 2. Instantiate Model and Optimizer
    model = BetaVAE(
        in_channels=in_channels,
        latent_dim=latent_dim,
        img_size=img_size,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    annealer = KLAnnealer(target_beta=target_beta, warmup_epochs=warmup_epochs)

    # 3. Checkpoint directory
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = checkpoint_dir / "best_vae_model.pt"

    best_val_recon = float("inf")

    # 4. Training Epochs
    for epoch in range(1, epochs + 1):
        beta = annealer.get_beta(epoch - 1)
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            loss_type=loss_type,
        )

        val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            beta=beta,
            loss_type=loss_type,
        )

        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Recon MSE: {train_metrics['recon_loss']:.5f} | "
            f"KLD: {train_metrics['kld_loss']:.2f} | "
            f"Val Recon MSE: {val_metrics['val_recon_loss']:.5f} | "
            f"Beta: {beta:.6f}"
        )

        # Checkpoint best model according to healthy reconstruction error
        if val_metrics["val_recon_loss"] < best_val_recon:
            best_val_recon = val_metrics["val_recon_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_dim": latent_dim,
                    "img_size": img_size,
                    "in_channels": in_channels,
                    "best_val_recon": best_val_recon,
                },
                str(best_weights_path),
            )
            print(f"  --> Saved new best checkpoint (Val MSE: {best_val_recon:.5f}) to {best_weights_path}")

    print(f"[*] Training complete. Best model saved at: {best_weights_path}")
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Beta-VAE on Brain MRI scans.")
    parser.add_argument("--data_dir", type=str, default="data/brain_mri")
    parser.add_argument("--output_dir", type=str, default="backend/checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5)

    args = parser.parse_args()
    train_vae(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        target_beta=args.beta,
        warmup_epochs=args.warmup,
    )
