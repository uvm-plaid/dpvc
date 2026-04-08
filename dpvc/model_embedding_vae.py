import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from . import utils


class Encoder(nn.Module):
    def __init__(self, input_dim=256, latent_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 64),
            nn.GELU(),
        )

        self.to_mu = nn.Linear(64, latent_dim)
        self.to_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=6, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),

            nn.Linear(64, 256),
            nn.GELU(),

            nn.Linear(256, 512),
            nn.GELU(),

            nn.Linear(512, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=256, latent_dims=6,
                 clip_threshold=10.0,
                 post_clip_threshold=10.0):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dims)
        self.decoder = Decoder(latent_dims, input_dim)
        self.noise_mult = None
        self.clip_threshold = clip_threshold
        self.post_clip_threshold = post_clip_threshold

    def set_noise_mult(self, noise_mult):
        self.noise_mult = noise_mult

    def clip_by_l2norm(self, x, threshold):
        norms = x.norm(p=2, dim=1, keepdim=True)
        scaling = torch.clamp(threshold / (norms + 1e-8), max=1.0)
        return x * scaling

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, seed=None, control_features=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.last_z = z

        if self.noise_mult:
            z = self.clip_by_l2norm(z, self.clip_threshold)

            utils.set_seed(seed)
            z = z + self.clip_threshold * self.noise_mult * torch.randn(z.shape).to(z.device)

            z = self.clip_by_l2norm(z, 2 * self.clip_threshold)
            z = torch.clamp(z, min=-self.post_clip_threshold, max=self.post_clip_threshold)

        if control_features:
            assert isinstance(control_features, dict)
            for idx, val in control_features.items():
                z[:, idx] = val

        recon = self.decoder(z)
        self.kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon
