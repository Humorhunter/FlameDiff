"""Multi-scale Attention U-Net denoiser for spatial latent diffusion.

Directly inspired by Wu, Liang & Luo (Combustion and Flame 282, 2025, Table 2):
4-level hierarchical ResNet U-Net with GroupNorm, SiLU, sinusoidal timestep
FiLM modulation, and multi-head self-attention at coarse resolutions
(12x16 and 6x8 for 48x64 latents).

Exposes the exact same call signature as ``ViTDecoder``:
``forward(x, y=None, t=timesteps) -> eps`` so it plugs seamlessly into
``diffusion_loss``, ``sample_mmps``, ``sample_adaptive_dps``, and
``TemperatureReconstructor``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        t = timesteps.reshape(-1).to(dtype=dtype)
        half = self.dim // 2
        if half == 0:
            return t.unsqueeze(-1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class _SelfAttention2d(nn.Module):
    """Explicit SDPA/matmul multi-head self-attention (forward-AD & float64 safe)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, self.head_dim, n)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, head_dim, N]
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q.transpose(-2, -1), k) * scale  # [B, heads, N, N]
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1)).reshape(b, c, h, w)
        return x + self.proj(out)


class _Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class LatentUNet(nn.Module):
    """Multi-scale Attention U-Net epsilon predictor on ``[B, C_lat, 48, 64]`` latents."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 1, 2, 3),
        num_res_blocks: int = 2,
        attention_levels: Sequence[int] = (2, 3),
        num_heads: int = 4,
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(m) for m in channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_levels = tuple(int(lvl) for lvl in attention_levels)
        self.num_heads = int(num_heads)

        time_dim = self.base_channels * 4
        self.time_sin = SinusoidalTimeEmbedding(self.base_channels)
        self.time_embed = nn.Sequential(
            nn.Linear(self.base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(self.in_channels, self.base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        skip_channels: list[int] = [self.base_channels]
        ch = self.base_channels
        attn_set = set(self.attention_levels)

        for level, mult in enumerate(self.channel_mults):
            out_ch = self.base_channels * mult
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                level_blocks.append(_ResBlock(ch, out_ch, time_dim))
                level_attns.append(
                    _SelfAttention2d(out_ch, num_heads=self.num_heads)
                    if level in attn_set
                    else nn.Identity()
                )
                ch = out_ch
                skip_channels.append(ch)
            self.down_blocks.append(level_blocks)
            self.down_attns.append(level_attns)
            if level < len(self.channel_mults) - 1:
                self.downsamplers.append(_Downsample(ch))
                skip_channels.append(ch)
            else:
                self.downsamplers.append(nn.Identity())

        # Bottleneck
        self.mid_block1 = _ResBlock(ch, ch, time_dim)
        self.mid_attn = _SelfAttention2d(ch, num_heads=self.num_heads)
        self.mid_block2 = _ResBlock(ch, ch, time_dim)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level in reversed(range(len(self.channel_mults))):
            out_ch = self.base_channels * self.channel_mults[level]
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                level_blocks.append(_ResBlock(ch + skip_ch, out_ch, time_dim))
                level_attns.append(
                    _SelfAttention2d(out_ch, num_heads=self.num_heads)
                    if level in attn_set
                    else nn.Identity()
                )
                ch = out_ch
            self.up_blocks.append(level_blocks)
            self.up_attns.append(level_attns)
            if level > 0:
                self.upsamplers.append(_Upsample(ch))
            else:
                self.upsamplers.append(nn.Identity())

        self.out_norm = _group_norm(ch)
        self.out_conv = nn.Conv2d(ch, self.in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del y  # Unconditional prior; matches ViTDecoder(x, y, t) interface
        if t is None:
            t = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        t_emb = self.time_embed(self.time_sin(t, dtype=x.dtype))

        h = self.in_conv(x)
        skips = [h]
        for level, (blocks, attns, down) in enumerate(
            zip(self.down_blocks, self.down_attns, self.downsamplers)
        ):
            for block, attn in zip(blocks, attns):
                h = attn(block(h, t_emb))
                skips.append(h)
            if level < len(self.channel_mults) - 1:
                h = down(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for blocks, attns, up in zip(self.up_blocks, self.up_attns, self.upsamplers):
            for block, attn in zip(blocks, attns):
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = attn(block(torch.cat([h, skip], dim=1), t_emb))
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
