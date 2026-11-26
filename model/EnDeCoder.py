import torch
from torch import nn
from torch.nn import functional as F


def _gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 1,
        in_channels: int = 1,
        latent_activation: str = "tanh",
    ):
        """Legacy AutoEncoder: downsample (in_channels, 192, 256) -> (latent_channels, 48, 64)."""
        super().__init__()
        if not isinstance(latent_channels, int) or isinstance(latent_channels, bool) or latent_channels < 1:
            raise ValueError("latent_channels must be a positive integer")
        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if latent_activation not in ("tanh", "linear"):
            raise ValueError("latent_activation must be 'tanh' or 'linear'")
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.latent_activation = latent_activation
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)  # (C_in, 192, 256) -> (64, 192, 256)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)           # (64, 192, 256) -> (64, 192, 256)
        self.pool1 = nn.MaxPool2d(2, 2)                        # (64, 192, 256) -> (64, 96, 128)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)          # (64, 96, 128) -> (128, 96, 128)
        self.conv4 = nn.Conv2d(128, 128, 5, padding=2)         # (128, 96, 128) -> (128, 96, 128)
        self.pool2 = nn.MaxPool2d(2, 2)                        # (128, 96, 128) -> (128, 48, 64)
        self.proj1 = nn.Linear(128, 64)                        # (128, 48, 64) -> (64, 48, 64)
        self.proj2 = nn.Linear(64, latent_channels)            # (64, 48, 64) -> (C_lat, 48, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = self.pool1(x)
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 3, 1)
        x = F.selu(self.proj1(x))
        x = self.proj2(x)
        if self.latent_activation == "tanh":
            x = torch.tanh(x)
        x = x.permute(0, 3, 1, 2)
        return x


class AutoDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 1,
        output_activation: str = "relu",
        out_channels: int = 1,
    ):
        """Legacy AutoDecoder: upsample (latent_channels, 48, 64) -> (out_channels, 192, 256)."""
        super().__init__()
        if not isinstance(latent_channels, int) or isinstance(latent_channels, bool) or latent_channels < 1:
            raise ValueError("latent_channels must be a positive integer")
        if not isinstance(out_channels, int) or isinstance(out_channels, bool) or out_channels < 1:
            raise ValueError("out_channels must be a positive integer")
        if output_activation not in ("relu", "linear"):
            raise ValueError("output_activation must be 'relu' or 'linear'")
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.in_channels = out_channels
        self.output_activation = output_activation
        self.proj1 = nn.Linear(latent_channels, 64)            # (C_lat, 48, 64) -> (64, 48, 64)
        self.proj2 = nn.Linear(64, 128)                        # (64, 48, 64) -> (128, 48, 64)
        self.conv5 = nn.Conv2d(128, 128, 5, padding=2)         # (128, 96, 128) -> (128, 96, 128)
        self.conv6 = nn.Conv2d(128, 64, 3, padding=1)          # (128, 96, 128) -> (64, 96, 128)
        self.conv7 = nn.Conv2d(64, 64, 5, padding=2)           # (64, 192, 256) -> (64, 192, 256)
        self.conv8 = nn.Conv2d(64, out_channels, 3, padding=1) # (64, 192, 256) -> (C_out, 192, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.selu(self.proj1(x))
        x = F.selu(self.proj2(x))
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=2)
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))
        x = F.interpolate(x, scale_factor=2)
        if self.output_activation == "linear":
            x = F.silu(self.conv7(x))
            x = self.conv8(x)
        else:
            x = F.relu(self.conv7(x))
            x = F.relu(self.conv8(x))
        return x


class AutoEnDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int = 1,
        output_activation: str = "relu",
        in_channels: int = 1,
        latent_activation: str = "tanh",
    ):
        """Legacy AutoEnDecoder: (in_channels, 192, 256) -> (latent_channels, 48, 64) -> (in_channels, 192, 256)."""
        super().__init__()
        self.arch = "legacy"
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.output_activation = output_activation
        self.latent_activation = latent_activation
        self.encoder = AutoEncoder(
            latent_channels=latent_channels,
            in_channels=in_channels,
            latent_activation=latent_activation,
        )
        self.decoder = AutoDecoder(
            latent_channels=latent_channels,
            output_activation=output_activation,
            out_channels=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _MultiScaleBlock(nn.Module):
    """Parallel 3x3 / 5x5 / 7x7 multi-scale conv block with 1x1 cross-channel fusion.

    Modeled on Wu, Liang & Luo (Combustion and Flame 282, 2025, Section 2.3, Fig. 2):
    parallel kernels capture local reaction fronts (3x3) and wider jet structures
    (5x5, 7x7), fused with GroupNorm + SiLU and a 1x1 projection.
    """

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1):
        super().__init__()
        branch_ch = max(out_ch // 3, 8)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, stride=stride, padding=1),
            _gn(branch_ch),
            nn.SiLU(),
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=5, stride=stride, padding=2),
            _gn(branch_ch),
            nn.SiLU(),
        )
        self.b7 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=7, stride=stride, padding=3),
            _gn(branch_ch),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, out_ch, kernel_size=1),
            _gn(out_ch),
            nn.SiLU(),
        )
        if stride == 1 and in_ch == out_ch:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return self.fuse(cat) + self.residual(x)


class _SubpixelUpsample(nn.Module):
    """PixelShuffle 2x upsampling avoiding transposed-conv checkerboard & bilinear blur."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = _gn(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.shuffle(self.conv(x))))


class MultiScaleEncoder(nn.Module):
    """MSCAE encoder: (in_channels, 192, 256) -> (latent_channels, 48, 64)."""

    def __init__(
        self,
        latent_channels: int = 1,
        in_channels: int = 3,
        base_channels: int = 48,
        latent_activation: str = "linear",
    ):
        super().__init__()
        if latent_channels < 1 or in_channels < 1:
            raise ValueError("latent_channels and in_channels must be positive integers")
        if latent_activation not in ("tanh", "linear"):
            raise ValueError("latent_activation must be 'tanh' or 'linear'")
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.latent_activation = latent_activation

        c1 = self.base_channels
        c2 = self.base_channels * 2
        self.stem = _MultiScaleBlock(self.in_channels, c1, stride=1)
        self.down1 = _MultiScaleBlock(c1, c1, stride=2)     # 96 x 128
        self.mid1 = _MultiScaleBlock(c1, c2, stride=1)
        self.down2 = _MultiScaleBlock(c2, c2, stride=2)     # 48 x 64
        self.mid2 = _MultiScaleBlock(c2, c2, stride=1)
        self.to_latent = nn.Conv2d(c2, self.latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.down1(h)
        h = self.mid1(h)
        h = self.down2(h)
        h = self.mid2(h)
        z = self.to_latent(h)
        if self.latent_activation == "tanh":
            z = torch.tanh(z)
        return z


class MultiScaleDecoder(nn.Module):
    """MSCAE decoder: (latent_channels, 48, 64) -> (out_channels, 192, 256)."""

    def __init__(
        self,
        latent_channels: int = 1,
        out_channels: int = 3,
        base_channels: int = 48,
        output_activation: str = "linear",
    ):
        super().__init__()
        if latent_channels < 1 or out_channels < 1:
            raise ValueError("latent_channels and out_channels must be positive integers")
        if output_activation not in ("relu", "linear"):
            raise ValueError("output_activation must be 'relu' or 'linear'")
        self.latent_channels = int(latent_channels)
        self.out_channels = int(out_channels)
        self.in_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.output_activation = output_activation

        c1 = self.base_channels
        c2 = self.base_channels * 2
        self.from_latent = nn.Sequential(
            nn.Conv2d(self.latent_channels, c2, kernel_size=1),
            _gn(c2),
            nn.SiLU(),
        )
        self.mid1 = _MultiScaleBlock(c2, c2, stride=1)
        self.up1 = _SubpixelUpsample(c2, c1)                # 96 x 128
        self.mid2 = _MultiScaleBlock(c1, c1, stride=1)
        self.up2 = _SubpixelUpsample(c1, c1)                # 192 x 256
        self.refine = _MultiScaleBlock(c1, c1, stride=1)
        self.out_conv = nn.Conv2d(c1, self.out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.mid1(h)
        h = self.up1(h)
        h = self.mid2(h)
        h = self.up2(h)
        h = self.refine(h)
        out = self.out_conv(h)
        if self.output_activation == "relu":
            out = F.relu(out)
        return out


class MultiScaleAutoEnDecoder(nn.Module):
    """Multi-scale convolutional autoencoder (MSCAE) with subpixel upsampling."""

    def __init__(
        self,
        latent_channels: int = 1,
        output_activation: str = "linear",
        in_channels: int = 3,
        latent_activation: str = "linear",
        base_channels: int = 48,
    ):
        super().__init__()
        self.arch = "mscae"
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.output_activation = output_activation
        self.latent_activation = latent_activation
        self.base_channels = int(base_channels)
        self.encoder = MultiScaleEncoder(
            latent_channels=self.latent_channels,
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            latent_activation=self.latent_activation,
        )
        self.decoder = MultiScaleDecoder(
            latent_channels=self.latent_channels,
            out_channels=self.in_channels,
            base_channels=self.base_channels,
            output_activation=self.output_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_autoencoder(ae_config: dict) -> nn.Module:
    """Instantiate either legacy ``AutoEnDecoder`` or ``MultiScaleAutoEnDecoder``."""
    arch = str(ae_config.get("arch", "legacy"))
    in_channels = int(ae_config.get("in_channels", 1))
    latent_channels = int(ae_config.get("latent_channels", 1))
    output_activation = str(ae_config.get("output_activation", "relu"))
    latent_activation = str(ae_config.get("latent_activation", "tanh"))
    if arch == "legacy":
        return AutoEnDecoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            output_activation=output_activation,
            latent_activation=latent_activation,
        )
    if arch == "mscae":
        return MultiScaleAutoEnDecoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            output_activation=output_activation,
            latent_activation=latent_activation,
            base_channels=int(ae_config.get("base_channels", 48)),
        )
    raise ValueError(f"Unsupported autoencoder arch {arch!r}; expected 'legacy' or 'mscae'")
