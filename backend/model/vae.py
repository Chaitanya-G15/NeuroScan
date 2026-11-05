"""Variational Autoencoder (VAE) implementation for Unsupervised Medical Anomaly Detection.

Contains modular definitions for:
- Encoder: Compresses high-dimensional medical images to latent parameter vectors (mu, logvar).
- Reparameterization Bottleneck: Enables backpropagation through stochastic sampling.
- Decoder: Reconstructs healthy anatomical structures from latent space vector z.
- BetaVAE: Full model wrapper with explicit forward pass shape tracking.
- vae_loss_function: Computes reconstruction loss (MSE/BCE) combined with scaled KL divergence.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional Encoder projecting input medical scans into mean and log-variance latent representations."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (32, 64, 128, 256),
        img_size: int = 128,
    ):
        """Initializes the Encoder layers.

        Args:
            in_channels (int): Number of image input channels (1 for grayscale MRI/CT, 3 for RGB).
            latent_dim (int): Dimensionality of the latent bottleneck vector z.
            hidden_dims (Tuple[int, ...]): Channel capacities for successive Conv2d layers.
            img_size (int): Spatial height/width of input images (must be divisible by 2**len(hidden_dims)).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        layers = []
        prev_channels = in_channels
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=prev_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            prev_channels = h_dim

        self.conv_net = nn.Sequential(*layers)

        # Calculate flattened spatial size after downsamplings
        num_downsamples = len(hidden_dims)
        self.feature_map_size = img_size // (2**num_downsamples)
        self.flatten_dim = hidden_dims[-1] * (self.feature_map_size**2)

        # Latent heads: Mean and Log-Variance
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes an input tensor into mu and log-variance vectors.

        Args:
            x (torch.Tensor): Input medical scans. Shape: [B, in_channels, H, W]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mu: Mean vector of shape [B, latent_dim]
                - logvar: Log-variance vector of shape [B, latent_dim]
        """
        # Input shape: [B, C, H, W]
        features = self.conv_net(x)
        # Shape after conv_net: [B, hidden_dims[-1], H / 2^k, W / 2^k]

        flattened = torch.flatten(features, start_dim=1)
        # Shape: [B, flatten_dim]

        mu = self.fc_mu(flattened)  # [B, latent_dim]
        logvar = self.fc_logvar(flattened)  # [B, latent_dim]

        return mu, logvar


class Decoder(nn.Module):
    """Convolutional Decoder projecting latent vector z back to image space."""

    def __init__(
        self,
        latent_dim: int = 64,
        out_channels: int = 1,
        hidden_dims: Tuple[int, ...] = (256, 128, 64, 32),
        feature_map_size: int = 8,
    ):
        """Initializes the Decoder layers.

        Args:
            latent_dim (int): Dimensionality of the latent bottleneck vector z.
            out_channels (int): Number of output image channels (1 for grayscale, 3 for RGB).
            hidden_dims (Tuple[int, ...]): Channel capacities for ConvTranspose2d layers (in reverse order).
            feature_map_size (int): Spatial height/width of initial feature map before upsampling.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_map_size = feature_map_size
        self.initial_channels = hidden_dims[0]

        self.fc_project = nn.Sequential(
            nn.Linear(latent_dim, self.initial_channels * (feature_map_size**2)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        # Final upsampling layer to original resolution and channel depth
        layers.extend(
            [
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[-1],
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.Sigmoid(),  # Bound output strictly to [0.0, 1.0] for pixel intensities
            ]
        )

        self.deconv_net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent representations back into reconstructed images.

        Args:
            z (torch.Tensor): Latent vector. Shape: [B, latent_dim]

        Returns:
            torch.Tensor: Reconstructed image. Shape: [B, out_channels, H, W]
        """
        # Shape: [B, latent_dim]
        projected = self.fc_project(z)
        # Shape: [B, hidden_dims[0] * feature_map_size * feature_map_size]

        reshaped = projected.view(
            -1, self.initial_channels, self.feature_map_size, self.feature_map_size
        )
        # Shape: [B, hidden_dims[0], feature_map_size, feature_map_size]

        reconstructed = self.deconv_net(reshaped)
        # Shape: [B, out_channels, H, W]

        return reconstructed


class BetaVAE(nn.Module):
    """Complete VAE Architecture for Unsupervised Anomaly Detection."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (32, 64, 128, 256),
        img_size: int = 128,
    ):
        """Initializes the BetaVAE model.

        Args:
            in_channels (int): Input image channels (default: 1 for grayscale MRI/CT).
            latent_dim (int): Latent space dimension.
            hidden_dims (Tuple[int, ...]): Channel progression for encoder/decoder.
            img_size (int): Image dimension (height and width).
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            img_size=img_size,
        )

        decoder_hidden_dims = tuple(reversed(hidden_dims))
        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            hidden_dims=decoder_hidden_dims,
            feature_map_size=self.encoder.feature_map_size,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick to sample z from N(mu, exp(logvar)).

        Args:
            mu (torch.Tensor): Mean vector of shape [B, latent_dim].
            logvar (torch.Tensor): Log-variance vector of shape [B, latent_dim].

        Returns:
            torch.Tensor: Sampled latent vector z of shape [B, latent_dim].
        """
        if self.training:
            # Clamp logvar to prevent extreme variance instabilities
            clamped_logvar = torch.clamp(logvar, min=-15.0, max=10.0)
            std = torch.exp(0.5 * clamped_logvar)
            eps = torch.randn_like(std)  # eps ~ N(0, I)
            return mu + (eps * std)
        # During evaluation/inference, deterministically use the mean vector
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs full forward pass: Encode -> Reparameterize -> Decode.

        Args:
            x (torch.Tensor): Input medical image batch. Shape: [B, C, H, W]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - recon_x: Reconstructed batch [B, C, H, W]
                - mu: Latent means [B, latent_dim]
                - logvar: Latent log-variances [B, latent_dim]
        """
        # [B, C, H, W] -> mu: [B, latent_dim], logvar: [B, latent_dim]
        mu, logvar = self.encoder(x)

        # [B, latent_dim] -> [B, latent_dim]
        z = self.reparameterize(mu, logvar)

        # [B, latent_dim] -> [B, C, H, W]
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Generates clean reconstruction directly for inference.

        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W].

        Returns:
            torch.Tensor: Reconstructed image tensor [B, C, H, W].
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
            return self.decoder(mu)


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4,
    loss_type: str = "mse",
) -> Dict[str, torch.Tensor]:
    """Computes Beta-VAE loss: Mean Reconstruction Loss (per pixel) + beta * Mean KL Divergence.

    Formulated on mean pixel scale to maintain proper balance between reconstruction
    quality (learning intricate brain sulci/ventricles) and Gaussian prior regularization.

    Args:
        recon_x (torch.Tensor): Reconstructed images [B, C, H, W].
        x (torch.Tensor): Ground-truth input images [B, C, H, W].
        mu (torch.Tensor): Latent distribution mean [B, latent_dim].
        logvar (torch.Tensor): Latent distribution log-variance [B, latent_dim].
        beta (float): KL weighting hyperparameter. Default: 1e-4.
        loss_type (str): "mse" or "bce" for reconstruction loss calculation.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - 'loss': Total combined weighted loss scalar.
            - 'recon_loss': Mean reconstruction loss scalar.
            - 'kld_loss': Mean KL divergence loss scalar.
    """
    # 1. Reconstruction Loss (mean per pixel across batch and channels)
    if loss_type.lower() == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")
    else:
        # MSE per pixel in range [0, 1]
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    # 2. Analytic KL Divergence between N(mu, sigma^2) and N(0, I)
    # Sum over latent dimensions, average over batch
    clamped_logvar = torch.clamp(logvar, min=-15.0, max=10.0)
    kld_loss = -0.5 * torch.sum(1 + clamped_logvar - mu.pow(2) - torch.exp(clamped_logvar), dim=-1).mean()

    # 3. Total Balanced Loss
    total_loss = recon_loss + (beta * kld_loss)

    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "kld_loss": kld_loss,
    }
