"""Linear observation operators for spatial latent-MMPS reconstruction.

All image tensors use ``[batch, 1, height, width]`` and temperature values in
the same units/normalization as the autoencoder input.  The operators are
linear in the image and expose ``measure`` (A) and its true adjoint
``adjoint`` (A^T), which MMPS can compose with the latent decoder.

The default sensor coordinates are copied from ``config/default_training.yaml``.
They are listed as ``(x, y)`` and measurements retain that exact list order.
Inpainting observations keep the image shape and set the missing rectangle to
zero; NaNs in that rectangle are also accepted by ``validate_observation`` and
ignored by the adjoint/projection.  SR observations are non-overlapping 4x4
block means.
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
    """Task-agnostic spatial measurement operator.

    Args:
        task: ``"sensors"``, ``"inpaint"``, or ``"sr"``.
        sensor_coords_xy: Exactly eight distinct ``(x, y)`` integer locations.
            Defaults to the coordinates in ``config/default_training.yaml``.

    The current benchmark geometry is fixed at ``[1, 192, 256]``.  Inpainting
    uses the existing half-open rectangle ``[48:144, 64:192]`` and SR uses 4x4
    block means.

    ``measure`` always returns a batch-first tensor.  Finite-value validation
    is deliberately explicit via ``validate_field`` / ``validate_observation``
    so iterative GPU samplers do not synchronize the device at every ``A(x)``.
    Shape and dtype checks remain active on every public operation.
    """

    FIELD_SHAPE = (1, 192, 256)
    SCALE = 4
    HOLE_BOUNDS = (48, 144, 64, 192)

    def __init__(
        self,
        task: str,
        *,
        sensor_coords_xy: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if not isinstance(task, str) or task not in {"sensors", "inpaint", "sr"}:
            raise ValueError("task must be one of: sensors, inpaint, sr")
        self.task = task

        self.field_shape = self.FIELD_SHAPE
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

        # Flattened indices use standard row-major image order, converting the
        # public/config (x, y) coordinates to index y * width + x.
        self._sensor_indices_cpu = tuple(y * self.width + x for x, y in self.sensor_coords_xy)
        self._sensor_indices_cache: dict[torch.device, torch.Tensor] = {}
        self._sensor_pixel_mask_cache: dict[torch.device, torch.Tensor] = {}
        self._base_mask_cache: dict[torch.device, torch.Tensor] = {}

    def _sensor_indices(self, device: torch.device) -> torch.Tensor:
        indices = self._sensor_indices_cache.get(device)
        if indices is None:
            indices = torch.tensor(self._sensor_indices_cpu, dtype=torch.long, device=device)
            self._sensor_indices_cache[device] = indices
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

    def _base_mask(self, device: torch.device) -> torch.Tensor:
        mask = self._base_mask_cache.get(device)
        if mask is None:
            mask = torch.ones((1, 1, self.height, self.width), dtype=torch.bool, device=device)
            if self.task == "inpaint":
                mask[..., self.y0:self.y1, self.x0:self.x1] = False
            elif self.task == "sensors":
                mask.zero_()
                ys = torch.tensor([y for _, y in self.sensor_coords_xy], dtype=torch.long, device=device)
                xs = torch.tensor([x for x, _ in self.sensor_coords_xy], dtype=torch.long, device=device)
                mask[:, 0, ys, xs] = True
            self._base_mask_cache[device] = mask
        return mask

    @property
    def shape(self) -> tuple[int, ...]:
        """Observation shape excluding batch."""
        if self.task == "sensors":
            return (8,)
        if self.task == "inpaint":
            return self.field_shape
        return (1, self.height // self.scale, self.width // self.scale)

    @property
    def measured_count(self) -> int:
        """Number of scalar measurements per sample."""
        if self.task == "sensors":
            return 8
        if self.task == "inpaint":
            return self.height * self.width - (self.y1 - self.y0) * (self.x1 - self.x0)
        return (self.height // self.scale) * (self.width // self.scale)

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

    def validate_observation(
        self,
        observation: torch.Tensor,
        *,
        batch_size: int | None = None,
        reference: torch.Tensor | None = None,
        finite: bool = True,
    ) -> torch.Tensor:
        """Validate an observation; missing inpaint cells may contain NaN/Inf.

        ``reference`` can be supplied to also require matching batch, device,
        and dtype.  Inpaint values inside the unknown rectangle are ignored.
        """
        self._check_observation(observation, batch_size=batch_size)
        if reference is not None:
            self._check_image(reference)
            if observation.shape[0] != reference.shape[0]:
                raise ValueError("field and observation batch sizes must match")
            if observation.device != reference.device or observation.dtype != reference.dtype:
                raise ValueError("field and observation must have the same device and dtype")
        if finite:
            if self.task == "inpaint":
                known = self.mask(observation)
                values = observation.masked_select(known)
                is_finite = torch.isfinite(values).all()
            else:
                is_finite = torch.isfinite(observation).all()
            if not bool(is_finite.item()):
                raise ValueError("observation contains NaN or infinite measured values")
        return observation

    def mask(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Return image-space bool mask (True where the field is observed).

        For ``sensors`` only the eight configured pixels are True; for
        ``inpaint`` the mask is False in the missing rectangle; SR returns all
        True because every field cell contributes to a block measurement.
        The result has shape ``[B, 1, H, W]`` and follows ``reference`` device
        and batch when provided (otherwise batch size is one on CPU).
        """
        if reference is None:
            batch_size, device = 1, torch.device("cpu")
        else:
            self._check_image(reference)
            batch_size, device = reference.shape[0], reference.device

        return self._base_mask(device).expand(batch_size, -1, -1, -1).clone()

    def measure(self, field: torch.Tensor) -> torch.Tensor:
        """Apply ``A`` to a batch of fields while preserving gradients."""
        self._check_image(field)
        if self.task == "sensors":
            indices = self._sensor_indices(field.device)
            return field[:, 0].flatten(1).index_select(1, indices)
        if self.task == "inpaint":
            zero = torch.zeros((), dtype=field.dtype, device=field.device)
            return torch.where(self._base_mask(field.device), field, zero)
        return F.avg_pool2d(field, kernel_size=self.scale, stride=self.scale)

    def adjoint(self, observation: torch.Tensor) -> torch.Tensor:
        """Apply the exact Euclidean adjoint ``A^T`` in image space."""
        self._check_observation(observation)
        batch_size = observation.shape[0]
        if self.task == "sensors":
            flat = torch.zeros(
                (batch_size, self.height * self.width),
                dtype=observation.dtype,
                device=observation.device,
            )
            indices = self._sensor_indices(observation.device)
            flat = flat.scatter(1, indices.expand(batch_size, -1), observation)
            return flat.reshape(batch_size, 1, self.height, self.width)
        if self.task == "inpaint":
            zero = torch.zeros((), dtype=observation.dtype, device=observation.device)
            return torch.where(self._base_mask(observation.device), observation, zero)
        repeated = observation.repeat_interleave(self.scale, dim=-2)
        repeated = repeated.repeat_interleave(self.scale, dim=-1)
        return repeated / (self.scale**2)

    def project(self, field: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Project ``field`` onto ``A(x) = observation`` when feasible.

        This is the orthogonal affine projection for each operator: overwrite
        sensor/known pixels, or distribute each SR block-mean discrepancy
        equally over that block.  Projection remains differentiable in both
        arguments.  The caller should call ``validate_observation`` once before
        an iterative sampling loop; missing inpaint values are ignored.
        """
        self._check_image(field)
        self._check_observation(observation, batch_size=field.shape[0])
        if observation.device != field.device or observation.dtype != field.dtype:
            raise ValueError("field and observation must have the same device and dtype")

        if self.task == "sensors":
            flat_field = field.flatten(2)
            indices = self._sensor_indices(field.device)
            pixel_mask = self._sensor_pixel_mask(field.device)
            measured = torch.zeros_like(flat_field).scatter(
                2, indices.view(1, 1, -1).expand(field.shape[0], -1, -1), observation.unsqueeze(1)
            )
            return torch.where(pixel_mask, measured, flat_field).reshape_as(field)
        if self.task == "inpaint":
            return torch.where(self._base_mask(field.device), observation, field)

        discrepancy = observation - self.measure(field)
        correction = discrepancy.repeat_interleave(self.scale, dim=-2)
        correction = correction.repeat_interleave(self.scale, dim=-1)
        return field + correction

__all__ = ["DEFAULT_SENSOR_COORDS_XY", "SpatialObservation"]
