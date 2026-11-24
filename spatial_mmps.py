"""Matrix-free local-linearized MMPS guidance for spatial latent diffusion."""

from math import isfinite, sqrt

import torch
from torch.autograd import forward_ad


_GMRES_TOLERANCE = 1e-5


def _as_scalar_float(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a finite scalar")
        value = value.detach().item()
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite scalar") from None
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return value


def _require_finite(name, value):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"MMPS encountered non-finite values in {name}")


def _gmres(operator, rhs, maxiter, *, tolerance=_GMRES_TOLERANCE):
    """Solve a small matrix-free system with modified Gram-Schmidt GMRES."""
    shape = rhs.shape
    b = rhs.reshape(-1)
    beta = torch.linalg.vector_norm(b)
    _require_finite("the GMRES right-hand side", beta)
    beta_value = float(beta.item())
    if beta_value == 0.0:
        return torch.zeros_like(rhs), {
            "relative_residual": 0.0,
            "iterations": 0.0,
            "converged": 1.0,
            "breakdown": 0.0,
        }

    limit = min(maxiter, b.numel())
    basis = [b / beta]
    hessenberg = torch.zeros(
        (limit + 1, limit), dtype=b.dtype, device=b.device
    )
    cosine = []
    sine = []
    g = torch.zeros(limit + 1, dtype=b.dtype, device=b.device)
    g[0] = beta
    iterations = 0
    breakdown = False

    for column in range(limit):
        vector = operator(basis[column].reshape(shape)).reshape(-1)
        _require_finite("a GMRES matrix-vector product", vector)
        unorthogonalized_norm = torch.linalg.vector_norm(vector)

        # Two MGS passes keep the tiny Krylov basis orthogonal in float32.
        for _ in range(2):
            for row in range(column + 1):
                coefficient = torch.dot(basis[row], vector)
                hessenberg[row, column] += coefficient
                vector = vector - coefficient * basis[row]

        next_norm = torch.linalg.vector_norm(vector)
        _require_finite("the GMRES Hessenberg matrix", hessenberg[:, column])
        _require_finite("the GMRES Arnoldi residual", next_norm)
        hessenberg[column + 1, column] = next_norm

        # Apply the existing Givens rotations to the new Hessenberg column.
        for row in range(column):
            upper = hessenberg[row, column].clone()
            lower = hessenberg[row + 1, column].clone()
            hessenberg[row, column] = cosine[row] * upper + sine[row] * lower
            hessenberg[row + 1, column] = -sine[row] * upper + cosine[row] * lower

        upper = hessenberg[column, column].clone()
        lower = hessenberg[column + 1, column].clone()
        radius = torch.hypot(upper, lower)
        _require_finite("the GMRES Givens rotation", radius)
        radius_value = float(radius.item())
        if radius_value == 0.0:
            # This Krylov direction cannot reduce the residual. Keep the best
            # solution from earlier iterations and report the breakdown.
            breakdown = True
            break

        c = upper / radius
        s = lower / radius
        cosine.append(c)
        sine.append(s)
        hessenberg[column, column] = radius
        hessenberg[column + 1, column] = 0

        old_g = g[column].clone()
        next_g = g[column + 1].clone()
        g[column] = c * old_g + s * next_g
        g[column + 1] = -s * old_g + c * next_g
        iterations = column + 1

        # Happy breakdown means the Krylov subspace is invariant. The current
        # Hessenberg system remains solvable and may already be exact.
        arnoldi_scale = max(float(unorthogonalized_norm.item()), torch.finfo(b.dtype).tiny)
        arnoldi_breakdown = float(next_norm.item()) <= torch.finfo(b.dtype).eps * arnoldi_scale
        if float((g[column + 1].abs() / beta).item()) <= tolerance or arnoldi_breakdown:
            breakdown = arnoldi_breakdown and float((g[column + 1].abs() / beta).item()) > tolerance
            break
        if column + 1 < limit:
            basis.append(vector / next_norm)

    if iterations == 0:
        solution = torch.zeros_like(b)
    else:
        triangular = hessenberg[:iterations, :iterations]
        target = g[:iterations]
        diagonal = torch.diagonal(triangular)
        if not torch.isfinite(diagonal).all().item() or (diagonal == 0).any().item():
            # Singular projected systems indicate a numerical breakdown. The
            # zero vector is safer than returning a NaN guidance direction.
            solution = torch.zeros_like(b)
            breakdown = True
            iterations = 0
        else:
            try:
                coefficients = torch.linalg.solve_triangular(
                    triangular, target[:, None], upper=True
                ).squeeze(1)
            except RuntimeError as error:
                raise FloatingPointError("MMPS GMRES projected solve failed") from error
            solution = torch.zeros_like(b)
            for index, coefficient in enumerate(coefficients):
                solution = solution + coefficient * basis[index]

    _require_finite("the GMRES solution", solution)
    actual_residual = b - operator(solution.reshape(shape)).reshape(-1)
    _require_finite("the GMRES residual", actual_residual)
    relative_residual = float((torch.linalg.vector_norm(actual_residual) / beta).item())
    return solution.reshape(shape), {
        "relative_residual": relative_residual,
        "iterations": float(iterations),
        "converged": float(relative_residual <= tolerance),
        "breakdown": float(breakdown),
    }


def mmps_guidance(
    current,
    clean,
    observation,
    measure,
    alpha_bar,
    *,
    sigma,
    maxiter,
):
    """Compute a local-linearized MMPS likelihood score without dense Jacobians.

    ``measure`` maps a clean latent to the observed field (decoder followed by
    the observation operator). For a VP epsilon model, the local covariance is
    applied as ``((1-alpha_bar)/sqrt(alpha_bar)) * J_mu.T``. The covariance is
    held fixed during the GMRES solve, as in MMPS; only first derivatives of
    the denoiser-derived ``clean`` tensor are used.
    """
    if not isinstance(current, torch.Tensor) or not isinstance(clean, torch.Tensor):
        raise ValueError("current and clean must be tensors")
    if current.ndim != 4 or tuple(clean.shape) != tuple(current.shape):
        raise ValueError("current and clean must have matching BCHW latent shapes")
    if current.shape[0] != 1:
        raise ValueError("MMPS currently requires batch size one")
    if not current.requires_grad or not clean.requires_grad:
        raise ValueError("current and clean must retain their autograd graph")
    if not callable(measure):
        raise ValueError("measure must be callable")
    if not isinstance(observation, torch.Tensor) or observation.ndim < 2:
        raise ValueError("observation must be a batched tensor")
    if observation.shape[0] != 1:
        raise ValueError("MMPS currently requires batch size one")
    if current.device != clean.device or current.device != observation.device:
        raise ValueError("current, clean, and observation must share a device")
    if current.dtype != clean.dtype or current.dtype != observation.dtype:
        raise ValueError("current, clean, and observation must share a dtype")
    if not current.is_floating_point():
        raise ValueError("MMPS expects real floating-point tensors")
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    sigma_value = _as_scalar_float("sigma", sigma)
    if sigma_value <= 0:
        raise ValueError("sigma must be strictly positive")
    alpha_value = _as_scalar_float("alpha_bar", alpha_bar)
    if not 0 < alpha_value <= 1:
        raise ValueError("alpha_bar must be in (0, 1]")
    covariance_scale = (1 - alpha_value) / sqrt(alpha_value)

    _require_finite("current latent", current)
    _require_finite("clean latent", clean)
    _require_finite("observation", observation)
    measured = measure(clean)
    if not isinstance(measured, torch.Tensor) or tuple(measured.shape) != tuple(observation.shape):
        raise ValueError("measure(clean) must return a tensor matching observation.shape")
    if measured.device != observation.device or measured.dtype != observation.dtype:
        raise ValueError("measure(clean) and observation must share device and dtype")
    _require_finite("measured observation", measured)
    rhs = observation - measured
    work_dtype = torch.float64 if rhs.dtype == torch.float64 else torch.float32

    def latent_vjp(tangent):
        tangent = tangent.to(dtype=clean.dtype)
        gradient, = torch.autograd.grad(
            clean,
            current,
            grad_outputs=tangent,
            retain_graph=True,
            create_graph=False,
        )
        return gradient

    def measurement_vjp(tangent):
        tangent = tangent.to(dtype=measured.dtype)
        gradient, = torch.autograd.grad(
            measured,
            clean,
            grad_outputs=tangent,
            retain_graph=True,
            create_graph=False,
        )
        return gradient

    def measurement_jvp(tangent):
        tangent = tangent.to(dtype=clean.dtype)
        with torch.no_grad(), forward_ad.dual_level():
            dual_input = forward_ad.make_dual(clean.detach(), tangent.detach())
            dual_output = measure(dual_input)
            result = forward_ad.unpack_dual(dual_output).tangent
            if result is None:
                return torch.zeros_like(dual_output)
            return result

    def covariance_measurement_product(vector):
        latent_vector = measurement_vjp(vector)
        propagated = covariance_scale * latent_vjp(latent_vector)
        return measurement_jvp(propagated)

    sigma_squared = sigma_value * sigma_value

    def system_product(vector):
        vector_work = vector.to(dtype=work_dtype)
        covariance_product = covariance_measurement_product(
            vector_work.to(dtype=measured.dtype)
        )
        return sigma_squared * vector_work + covariance_product.to(dtype=work_dtype)

    rhs_work = rhs.detach().to(dtype=work_dtype)
    _require_finite("the MMPS linear-system right-hand side", rhs_work)
    try:
        solved, diagnostics = _gmres(system_product, rhs_work, maxiter)
    except FloatingPointError:
        raise
    except RuntimeError as error:
        raise RuntimeError("MMPS matrix-free derivative or GMRES operation failed") from error

    # The posterior score is J_mu.T A.T v. Gradients are intentionally not
    # retained, so the sampler applies this as a detached epsilon correction.
    measurement_cotangent = measurement_vjp(solved.detach())
    likelihood_score = latent_vjp(measurement_cotangent).detach()
    _require_finite("the MMPS likelihood score", likelihood_score)
    diagnostics.update(
        {
            "measurement_rmse": float(torch.mean(rhs_work.square()).sqrt().item()),
            "sigma": float(sigma_value),
            "alpha_bar": float(alpha_value),
        }
    )
    return likelihood_score, diagnostics
