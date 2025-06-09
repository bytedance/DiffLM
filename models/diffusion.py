# Copyright (c) 2023 amazon-science
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/amazon-science/tabsyn/blob/main/tabsyn/model.py.
#
# This modified file is released under the same license.

import os
import json
import torch
import torch.nn as nn
import torch.optim
import numpy as np

from utils.module import SiLU, ReGLU, GEGLU
from utils.diffusion import EDMLoss, VELoss, VPLoss


DIFFUSION_MODEL_DIR = "diffusion_model"


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPBlock(nn.Module):
    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t = 512, num_layers = 1):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            MLPBlock(dim_t * 2, num_layers=num_layers),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

    def forward(self, x, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.mlp(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, dim)
        Returns:
            out: (batch_size, dim)
        """
        B, C = x.shape
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(1)

        # Self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Compute attention output
        out = (attn @ v).transpose(1, 2).reshape(B, C)
        out = self.proj(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=5, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                SelfAttention(dim, num_heads, dropout),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),  # Optional feed-forward
                nn.SiLU()
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        return x


class AttentionDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512, num_heads=8, num_layers=5):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        # Multi-layer Self-Attention block
        self.attention = nn.Sequential(
            nn.Linear(dim_t, dim_t),  # Initial projection
            nn.SiLU(),
            AttentionBlock(dim_t, num_heads=num_heads, num_layers=num_layers),  # Multi-layer Attention
            nn.Linear(dim_t, dim_t),  # Output projection
            nn.SiLU(),
            nn.Linear(dim_t, d_in),  # Back to input dimensions
        )

    def forward(self, x, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.attention(x)


class Precond(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        sigma_min = 0,                # Minimum supported noise level.
        sigma_max = float("inf"),     # Maximum supported noise level.
        sigma_data = 0.5,             # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):
        float_dtype = torch.float32

        x = x.to(float_dtype)
        sigma = sigma.to(float_dtype).reshape(-1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(float_dtype), c_noise.flatten())

        assert F_x.dtype == float_dtype
        D_x = c_skip * x + c_out * F_x.to(float_dtype)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class LatentDiffuser(nn.Module):
    def __init__(self, latent_size, dim_noise, denoising_impl="attn", denoising_layers=10, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5, opts=None, pfgmpp = False):
        super().__init__()
        if denoising_impl == "mlp":
            self.denoise_fn = MLPDiffusion(latent_size, dim_t=dim_noise, num_layers=denoising_layers)
        elif denoising_impl == "attn":
            self.denoise_fn = AttentionDiffusion(latent_size, dim_t=dim_noise, num_heads=4, num_layers=denoising_layers)
        else:
            raise NotImplementedError()
        self.precond_denoise_fn = Precond(self.denoise_fn, latent_size)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=latent_size, gamma=gamma, opts=opts)

    def forward(self, x):
        loss = self.loss_fn(self.precond_denoise_fn, x)
        return loss.mean(-1).mean()

    @staticmethod
    def load_from_saved_model(saved_model_path, device=None):
        config_file = os.path.join(saved_model_path, "config.json")
        best_ckpt_file = os.path.join(saved_model_path, "model_best.pt")

        with open(config_file, "r") as rf:
            config = json.load(rf)
        model = LatentDiffuser(config["latent_size"], config["dim_noise"], config["denoising_impl"], config["denoising_layers"])
        model.load_state_dict(torch.load(best_ckpt_file))

        train_file = os.path.join(config["vae_model_path"], "encodings", "train.npy")
        train_z = torch.tensor(np.load(train_file), dtype=torch.float)
        mean_z = train_z.mean(dim=0)

        if device is not None:
            model = model.to(device)
            mean_z = mean_z.to(device)

        return model, mean_z
