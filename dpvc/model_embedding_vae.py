import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from . import utils

INPUT_DIM = 256

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, INPUT_DIM)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = self.linear4(z)
        return z

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(INPUT_DIM, 128)
        self.linear11 = nn.Linear(128, 64)
        self.linear12 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, latent_dims)
        self.linear3 = nn.Linear(32, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear11(x))
        x = F.relu(self.linear12(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape).to(x.device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
        self.noise_mult = None
        self.clip_threshold = 5.0

    def set_noise_mult(self, noise_mult):
        self.noise_mult = noise_mult

    def clip_by_l2norm(self, x, threshold):
        norms = x.norm(p=2, dim=1, keepdim=True)
        scaling = torch.clamp(threshold / (norms + 1e-8), max=1.0)
        return x * scaling

    def forward(self, x, seed=None):
        z = self.encoder(x)
        if self.noise_mult:
            #print('******************* input max/min/norm:', x.max().item(), x.min().item(), x.norm(p=2).item())
            #print('******************* latent max/min/norm:', z.max().item(), z.min().item(), z.norm(p=2).item())
            z = self.clip_by_l2norm(z, self.clip_threshold)
            #print('******************* clipped max/min/norm:', z.max().item(), z.min().item(), z.norm(p=2).item())
            utils.set_seed(seed)
            z = z + self.clip_threshold*self.noise_mult*torch.randn(z.shape).to(z.device)
            #print('******************* noisy max/min/norm:', z.max().item(), z.min().item(), z.norm(p=2).item())
            z = torch.clamp(z, min=-self.clip_threshold, max=self.clip_threshold)
            #z = self.clip_by_l2norm(z, 10*self.clip_threshold)
        return self.decoder(z)

