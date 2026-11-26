"""Linear observation operators for single- and multi-field spatial latent-MMPS reconstruction.

All image tensors use ``[batch, num_channels, height, width]`` in the same
units/normalization as the autoencoder input.  The operators are linear in the
image and expose ``measure`` (A) and its exact Euclidean adjoint ``adjoint``
(A^T), which MMPS composes with the latent decoder.

Channel-selective observations (``observed_channels``) allow a single shared
multi-field prior (e.g. ``(T, OH, HO2)``) to solve both matched multi-channel
reconstruction and incomplete-physics cross-field inversion (e.g. observing
only ``T`` or only ``OH`` to infer unmeasured radical fields) without retraining.
"""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import torch
import torch.nn.functional as F


DEFAULT_SENSOR_COORDS_XY: tuple[tuple[int, int], ...] = (
    (20, 95),
    (70, 95),
    (110, 80),
    (110, 110),
    (160, 75),
    (160, 120),
    (200, 75),
    (200, 120),
)


class SpatialObservation:
    """Task-agnostic single- or multi-field spatial measurement operator.

    Args:
        task: ``"sensors"``, ``"inpaint"``, or ``"sr"``.
        sensor_coords_xy: Exactly eight distinct ``(x, y)`` integer locations.
            Defaults to the coordinates in ``config/default_training.yaml``.
        num_channels: Total number of physical channels in the full field
            (default ``1`` for single-channel temperature; ``3`` for ``(T, OH, HO2)``).
        observed_channels: Subset of channel indices ``[0, num_channels)`` that
            are actually observed. Defaults to all channels.
    """

    FIELD_SHAPE = (1, 192, 256)
    SCALE = 4
    HOLE_BOUNDS = (48, 144, 64, 192)

    def __init__(
        self,
        task: str,
        *,
        sensor_coords_xy: Sequence[Sequence[int]] | None = None,
        num_channels: int = 1,
        observed_channels: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(task, str) or task not in {"sensors", "inpaint", "sr"}:
            raise ValueError("task must be one of: sensors, inpaint, sr")
        if isinstance(num_channels, bool) or not isinstance(num_channels, Integral) or num_channels < 1:
            raise ValueError("num_channels must be a positive integer")
        self.task = task
        self.num_channels = int(num_channels)

        if observed_channels is None:
            self.observed_channels: tuple[int, ...] = tuple(range(self.num_channels))
        else:
            parsed = []
            for ch in observed_channels:
                if isinstance(ch, bool) or not isinstance(ch, Integral):
                    raise ValueError("observed_channels must contain integer channel indices")
                ch_int = int(ch)
                if not 0 <= ch_int < self.num_channels:
                    raise ValueError(f"observed channel index {ch_int} out of range [0, {self.num_channels})")
                if ch_int not in parsed:
                    parsed.append(ch_int)
            if not parsed:
                raise ValueError("observed_channels must contain at least one channel")
            self.observed_channels = tuple(parsed)
        self.num_observed_channels = len(self.observed_channels)

        self.field_shape = (self.num_channels, self.FIELD_SHAPE[1], self.FIELD_SHAPE[2])
        _, self.height, self.width = self.field_shape
        self.scale = self.SCALE
        self.y0, self.y1, self.x0, self.x1 = self.HOLE_BOUNDS

        coords_input = DEFAULT_SENSOR_COORDS_XY if sensor_coords_xy is None else sensor_coords_xy
        try:
            coords = tuple(tuple(point) for point in coords_input)
        except TypeError as exc:
            raise ValueError("sensor_coords_xy must contain eight (x, y) integer pairs") from exc
        if len(coords) != 8 or any(len(point) != 2 for point in coords):
            raise ValueError("sensor_coords_xy must contain exactly eight (x, y) pairs")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for point in coords
            for value in point
        ):
            raise ValueError("sensor coordinates must be integers")
        self.sensor_coords_xy = tuple((int(x), int(y)) for x, y in coords)
        if len(set(self.sensor_coords_xy)) != 8:
            raise ValueError("sensor coordinates must be distinct")
        if any(
            not (0 <= x < self.width and 0 <= y < self.height)
            for x, y in self.sensor_coords_xy
        ):
            raise ValueError("sensor coordinates must lie inside field_shape")

        self._sensor_indices_cpu = tuple(y * self.width + x for x, y in self.sensor_coords_xy)
        self._sensor_indices_cache: dict[torch.device, torch.Tensor] = {}
        self._obs_channel_indices_cache: dict[torch.device, torch.Tensor] = {}
        self._sensor_pixel_mask_cache: dict[torch.device, torch.Tensor] = {}
        self._base_mask_cache: dict[torch.device, torch.Tensor] = {}
        self._obs_spatial_mask_cache: dict[torch.device, torch.Tensor] = {}

    def _sensor_indices(self, device: torch.device) -> torch.Tensor:
        indices = self._sensor_indices_cache.get(device)
        if indices is None:
            indices = torch.tensor(self._sensor_indices_cpu, dtype=torch.long, device=device)
            self._sensor_indices_cache[device] = indices
        return indices

    def _obs_channel_indices(self, device: torch.device) -> torch.Tensor:
        indices = self._obs_channel_indices_cache.get(device)
        if indices is None:
            indices = torch.tensor(self.observed_channels, dtype=torch.long, device=device)
            self._obs_channel_indices_cache[device] = indices
        return indices

    def _sensor_pixel_mask(self, device: torch.device) -> torch.Tensor:
        pixel_mask = self._sensor_pixel_mask_cache.get(device)
        if pixel_mask is None:
            indices = self._sensor_indices(device)
            pixel_mask = torch.zeros(
                (1, 1, self.height * self.width), dtype=torch.bool, device=device
            ).scatter(2, indices.view(1, 1, -1), True)
            self._sensor_pixel_mask_cache[device] = pixel_mask
        return pixel_mask

    def _obs_spatial_mask(self, device: torch.device) -> torch.Tensor:
        """Spatial bool mask of shape [1, 1, H, W] for observed channels."""
        mask = self._obs_spatial_mask_cache.get(device)
        if mask is None:
            mask = torch.ones((1, 1, self.height, self.width), dtype=torch.bool, device=device)
            if self.task == "inpaint":
                mask[..., self.y0:self.y1, self.x0:self.x1] = False
            elif self.task == "sensors":
                mask.zero_()
                ys = torch.tensor([y for _, y in self.sensor_coords_xy], dtype=torch.long, device=device)
                xs = torch.tensor([x for x, _ in self.sensor_coords_xy], dtype=torch.long, device=device)
                mask[:, 0, ys, xs] = True
            self._obs_spatial_mask_cache[device] = mask
        return mask

    def _base_mask(self, device: torch.device) -> torch.Tensor:
        """Full-field bool mask of shape [1, num_channels, H, W]."""
        mask = self._base_mask_cache.get(device)
        if mask is None:
            spatial = self._obs_spatial_mask(device)
            mask = torch.zeros(
                (1, self.num_channels, self.height, self.width), dtype=torch.bool, device=device
            )
            for ch in self.observed_channels:
                mask[:, ch : ch + 1] = spatial
            self._base_mask_cache[device] = mask
        return mask

    @property
    def shape(self) -> tuple[int, ...]:
        """Observation shape excluding batch."""
        if self.task == "sensors":
            if self.num_channels == 1:
                return (8,)
            return (self.num_observed_channels, 8)
        if self.task == "inpaint":
            return (self.num_observed_channels, self.height, self.width)
        return (self.num_observed_channels, self.height // self.scale, self.width // self.scale)

    @property
    def measured_count(self) -> int:
        """Number of scalar measurements per sample."""
        if self.task == "sensors":
            return 8 * self.num_observed_channels
        if self.task == "inpaint":
            return (self.height * self.width - (self.y1 - self.y0) * (self.x1 - self.x0)) * self.num_observed_channels
        return (self.height // self.scale) * (self.width // self.scale) * self.num_observed_channels

    def _check_image(self, field: torch.Tensor) -> None:
        if not isinstance(field, torch.Tensor):
            raise TypeError("field must be a torch.Tensor")
        if field.ndim != 4 or tuple(field.shape[1:]) != self.field_shape:
            raise ValueError(
                f"field must have shape [B, {self.field_shape[0]}, "
                f"{self.height}, {self.width}]"
            )
        if field.shape[0] <= 0:
            raise ValueError("field batch must not be empty")
        if not torch.is_floating_point(field):
            raise TypeError("field must have a real floating-point dtype")

    def _check_observation(self, observation: torch.Tensor, batch_size: int | None = None) -> None:
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor")
        if observation.ndim != len(self.shape) + 1 or tuple(observation.shape[1:]) != self.shape:
            raise ValueError(f"observation must have shape [B, {', '.join(map(str, self.shape))}]")
        if observation.shape[0] <= 0:
            raise ValueError("observation batch must not be empty")
        if batch_size is not None and observation.shape[0] != batch_size:
            raise ValueError("field and observation batch sizes must match")
        if not torch.is_floating_point(observation):
            raise TypeError("observation must have a real floating-point dtype")

    def validate_field(self, field: torch.Tensor, *, finite: bool = True) -> torch.Tensor:
        """Validate an image, including finite values by default; return it."""
        self._check_image(field)
        if finite and not bool(torch.isfinite(field).all().item()):
            raise ValueError("field contains NaN or infinite values")
        return field

    def observation_mask(self, observation: torch.Tensor) -> torch.Tensor:
        """Return bool mask matching an inpainting observation's shape [B, C_obs, H, W]."""
        self._check_observation(observation)
        if self.task != "inpaint":
            return torch.ones_like(observation, dtype=torch.bool)
        return self._obs_spatial_mask(observation.device).expand(
            observation.shape[0], self.num_observed_channels, -1, -1
        ).clone()

    def validate_observation(
        self,
        observation: torch.Tensor,
        *,
        batch_size: int | None = None,
        reference: torch.Tensor | None = None,
        finite: bool = True,
    ) -> torch.Tensor:
        """Validate an observation; missing inpaint cells may contain NaN/Inf."""
        self._check_observation(observation, batch_size=batch_size)
        if reference is not None:
            self._check_image(reference)
            if observation.shape[0] != reference.shape[0]:
                raise ValueError("field and observation batch sizes must match")
            if observation.device != reference.device or observation.dtype != reference.dtype:
                raise ValueError("field and observation must have the same device and dtype")
        if finite:
            if self.task == "inpaint":
                known = self.observation_mask(observation)
                values = observation.masked_select(known)
                is_finite = torch.isfinite(values).all()
            else:
                is_finite = torch.isfinite(observation).all()
            if not bool(is_finite.item()):
                raise ValueError("observation contains NaN or infinite measured values")
        return observation

    def mask(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Return image-space bool mask (True where the field is observed).

        Returns shape ``[B, num_channels, H, W]``. Unobserved channels are
        False everywhere.
        """
        if reference is None:
            batch_size, device = 1, torch.device("cpu")
        else:
            self._check_image(reference)
            batch_size, device = reference.shape[0], reference.device

        return self._base_mask(device).expand(batch_size, -1, -1, -1).clone()

    def _select_observed_channels(self, field: torch.Tensor) -> torch.Tensor:
        if self.num_observed_channels == self.num_channels and self.observed_channels == tuple(range(self.num_channels)):
            return field
        return field.index_select(1, self._obs_channel_indices(field.device))

    def measure(self, field: torch.Tensor) -> torch.Tensor:
        """Apply ``A`` to a batch of fields while preserving gradients."""
        self._check_image(field)
        selected = self._select_observed_channels(field)
        if self.task == "sensors":
            indices = self._sensor_indices(field.device)
            values = selected.flatten(2).index_select(2, indices)  # [B, C_obs, 8]
            if self.num_channels == 1:
                return values[:, 0]
            return values
        if self.task == "inpaint":
            zero = torch.zeros((), dtype=field.dtype, device=field.device)
            return torch.where(self._obs_spatial_mask(field.device), selected, zero)
        return F.avg_pool2d(selected, kernel_size=self.scale, stride=self.scale)

    def _scatter_observed_channels(self, obs_full_spatial: torch.Tensor) -> torch.Tensor:
        if self.num_observed_channels == self.num_channels and self.observed_channels == tuple(range(self.num_channels)):
            return obs_full_spatial
        out = torch.zeros(
            (obs_full_spatial.shape[0], self.num_channels, self.height, self.width),
            dtype=obs_full_spatial.dtype,
            device=obs_full_spatial.device,
        )
        ch_idx = self._obs_channel_indices(obs_full_spatial.device).view(1, -1, 1, 1).expand_as(obs_full_spatial)
        return out.scatter(1, ch_idx, obs_full_spatial)

    def adjoint(self, observation: torch.Tensor) -> torch.Tensor:
        """Apply the exact Euclidean adjoint ``A^T`` in image space."""
        self._check_observation(observation)
        batch_size = observation.shape[0]
        if self.task == "sensors":
            obs_3d = observation.unsqueeze(1) if self.num_channels == 1 else observation
            flat = torch.zeros(
                (batch_size, self.num_observed_channels, self.height * self.width),
                dtype=observation.dtype,
                device=observation.device,
            )
            indices = self._sensor_indices(observation.device).view(1, 1, -1).expand(
                batch_size, self.num_observed_channels, -1
            )
            flat = flat.scatter(2, indices, obs_3d)
            spatial = flat.reshape(batch_size, self.num_observed_channels, self.height, self.width)
            return self._scatter_observed_channels(spatial)
        if self.task == "inpaint":
            zero = torch.zeros((), dtype=observation.dtype, device=observation.device)
            spatial = torch.where(self._obs_spatial_mask(observation.device), observation, zero)
            return self._scatter_observed_channels(spatial)
        repeated = observation.repeat_interleave(self.scale, dim=-2)
        repeated = repeated.repeat_interleave(self.scale, dim=-1)
        return self._scatter_observed_channels(repeated / (self.scale**2))

    def project(self, field: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Project ``field`` onto ``A(x) = observation`` on observed channels only."""
        self._check_image(field)
        self._check_observation(observation, batch_size=field.shape[0])
        if observation.device != field.device or observation.dtype != field.dtype:
            raise ValueError("field and observation must have the same device and dtype")

        if self.task == "sensors":
            obs_3d = observation.unsqueeze(1) if self.num_channels == 1 else observation
            indices = self._sensor_indices(field.device)
            pixel_mask = self._sensor_pixel_mask(field.device)
            selected = self._select_observed_channels(field).flatten(2)
            measured = torch.zeros_like(selected).scatter(
                2, indices.view(1, 1, -1).expand(field.shape[0], self.num_observed_channels, -1), obs_3d
            )
            updated_obs = torch.where(pixel_mask, measured, selected).reshape(
                field.shape[0], self.num_observed_channels, self.height, self.width
            )
        elif self.task == "inpaint":
            selected = self._select_observed_channels(field)
            updated_obs = torch.where(self._obs_spatial_mask(field.device), observation, selected)
        else:
            discrepancy = observation - self.measure(field)
            correction = discrepancy.repeat_interleave(self.scale, dim=-2)
            correction = correction.repeat_interleave(self.scale, dim=-1)
            updated_obs = self._select_observed_channels(field) + correction

        if self.num_observed_channels == self.num_channels and self.observed_channels == tuple(range(self.num_channels)):
            return updated_obs
        ch_idx = self._obs_channel_indices(field.device).view(1, -1, 1, 1).expand_as(updated_obs)
        return field.scatter(1, ch_idx, updated_obs)


__all__ = ["DEFAULT_SENSOR_COORDS_XY", "SpatialObservation"]
