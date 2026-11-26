"""Single-frame spatial reconstruction baselines for FlameDiff."""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SpatialUNet(nn.Module):
    """U-Net baseline for central inpainting or spatial super-resolution.

    Inputs and outputs use normalized single-channel tensors shaped ``[B,1,H,W]``.
    For inpainting, ``observation`` is a full-resolution image with its missing
    rectangle zeroed out. The operator supplies a boolean known-pixel mask.
    For super-resolution, ``observation`` is the low-resolution image; a bicubic
    resize is used as a base and the network predicts a residual.
    """

    def __init__(self, task: str, base_channels: int = 32):
        super().__init__()
        if task not in {"inpaint", "sr"}:
            raise ValueError("task must be 'inpaint' or 'sr'")
        self.task = task
        input_channels = 2 if task == "inpaint" else 1
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = _DoubleConv(input_channels, channels[0])
        self.enc2 = _DoubleConv(channels[0], channels[1])
        self.enc3 = _DoubleConv(channels[1], channels[2])
        self.enc4 = _DoubleConv(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(channels[3], channels[3] * 2)
        self.up4 = nn.ConvTranspose2d(channels[3] * 2, channels[3], 2, stride=2)
        self.dec4 = _DoubleConv(channels[3] * 2, channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = _DoubleConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = _DoubleConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = _DoubleConv(channels[0] * 2, channels[0])
        self.output = nn.Conv2d(channels[0], 1, kernel_size=1)

    def _network(self, x: torch.Tensor) -> torch.Tensor:
        first = self.enc1(x)
        second = self.enc2(self.pool(first))
        third = self.enc3(self.pool(second))
        fourth = self.enc4(self.pool(third))
        center = self.bottleneck(self.pool(fourth))
        decoded = self.dec4(torch.cat((self.up4(center), fourth), dim=1))
        decoded = self.dec3(torch.cat((self.up3(decoded), third), dim=1))
        decoded = self.dec2(torch.cat((self.up2(decoded), second), dim=1))
        decoded = self.dec1(torch.cat((self.up1(decoded), first), dim=1))
        return self.output(decoded)

    def forward(self, observation: torch.Tensor, operator: Any) -> torch.Tensor:
        if observation.ndim != 4 or observation.shape[1] != 1:
            raise ValueError("observation must have shape [B,1,H,W]")
        if getattr(operator, "task", self.task) != self.task:
            raise ValueError("operator task does not match this U-Net")

        if self.task == "inpaint":
            known = operator.mask(observation).to(device=observation.device, dtype=torch.bool)
            if known.shape != observation.shape:
                raise ValueError("operator mask must match the observation shape")
            prediction = self._network(torch.cat((observation, known.to(observation.dtype)), dim=1))
            return torch.where(known, observation, prediction)

        scale = int(operator.scale)
        if scale <= 0:
            raise ValueError("operator.scale must be positive")
        target_size = (observation.shape[-2] * scale, observation.shape[-1] * scale)
        baseline = F.interpolate(observation, size=target_size, mode="bicubic", align_corners=False)
        return baseline + self._network(baseline)


def interpolate(observation: torch.Tensor, operator: Any, method: str) -> torch.Tensor:
    """Return a classical interpolation baseline for the requested task.

    ``harmonic`` solves the discrete Laplace equation inside a central
    rectangular inpainting mask using DST-I per channel. For super-resolution,
    use ``bilinear`` or ``bicubic`` to resize to the operator's scale.
    """
    if observation.ndim != 4 or observation.shape[1] < 1:
        raise ValueError("observation must have shape [B,C,H,W] with C >= 1")
    task = getattr(operator, "task", None)

    if task == "sr":
        if method not in {"bilinear", "bicubic"}:
            raise ValueError("super-resolution interpolation must be bilinear or bicubic")
        scale = int(operator.scale)
        if scale <= 0:
            raise ValueError("operator.scale must be positive")
        kwargs = {"align_corners": False} if method in {"bilinear", "bicubic"} else {}
        return F.interpolate(observation, scale_factor=scale, mode=method, **kwargs)

    if task != "inpaint" or method != "harmonic":
        raise ValueError("inpainting interpolation must use method='harmonic'")
    if hasattr(operator, "observation_mask"):
        known = operator.observation_mask(observation).to(device=observation.device, dtype=torch.bool)
    else:
        known = operator.mask(observation).to(device=observation.device, dtype=torch.bool)
    if known.shape != observation.shape:
        raise ValueError("operator mask must match the observation shape")
    missing = ~known
    if not torch.all(missing.flatten(2).any(dim=2)):
        return observation.clone()

    result = observation.clone()
    for batch_index in range(result.shape[0]):
        for ch_index in range(result.shape[1]):
            hole = missing[batch_index, ch_index]
            rows, cols = torch.where(hole)
            top, bottom = int(rows.min()), int(rows.max()) + 1
            left, right = int(cols.min()), int(cols.max()) + 1
            rectangular = torch.zeros_like(hole)
            rectangular[top:bottom, left:right] = True
            if not torch.equal(hole, rectangular):
                raise ValueError("harmonic interpolation requires a rectangular missing region")
            if top == 0 or left == 0 or bottom == hole.shape[0] or right == hole.shape[1]:
                raise ValueError("harmonic interpolation requires known pixels around the hole")

            # Solve the five-point discrete Laplace equation in a separable
            # orthonormal DST-I basis, with the observed ring as Dirichlet data.
            patch = result[
                batch_index,
                ch_index,
                top - 1 : bottom + 1,
                left - 1 : right + 1,
            ]
            height, width = bottom - top, right - left
            rhs = torch.zeros((height, width), dtype=patch.dtype, device=patch.device)
            rhs[0, :] += patch[0, 1:-1]
            rhs[-1, :] += patch[-1, 1:-1]
            rhs[:, 0] += patch[1:-1, 0]
            rhs[:, -1] += patch[1:-1, -1]

            row_indices = torch.arange(1, height + 1, dtype=patch.dtype, device=patch.device)
            row_modes = torch.arange(1, height + 1, dtype=patch.dtype, device=patch.device)
            row_basis = torch.sin(
                torch.pi * row_indices[:, None] * row_modes[None, :] / (height + 1)
            ) * (2.0 / (height + 1)) ** 0.5
            col_indices = torch.arange(1, width + 1, dtype=patch.dtype, device=patch.device)
            col_modes = torch.arange(1, width + 1, dtype=patch.dtype, device=patch.device)
            col_basis = torch.sin(
                torch.pi * col_indices[:, None] * col_modes[None, :] / (width + 1)
            ) * (2.0 / (width + 1)) ** 0.5

            eigenvalues = (
                4.0 * torch.sin(torch.pi * row_modes / (2.0 * (height + 1))).square()[:, None]
                + 4.0 * torch.sin(torch.pi * col_modes / (2.0 * (width + 1))).square()[None, :]
            )
            transformed = row_basis.transpose(0, 1) @ rhs @ col_basis
            solution = row_basis @ (transformed / eigenvalues) @ col_basis.transpose(0, 1)
            result[batch_index, ch_index, top:bottom, left:right] = solution
    return result
