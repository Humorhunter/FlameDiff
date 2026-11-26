"""Relative observation guidance as an optional sampler class."""
import math

import torch


def relative_sampler(native_sampler, beta):
    """Build a sampler with the native DDPM proposal and relative correction."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Expected finite nonnegative beta.")

    class RelativeAmplitudeSampler(native_sampler):
        @torch.no_grad()
        def step(self, x_t, t, s, **kwargs):
            alpha_s, sigma_s = self.denoiser.schedule(s)
            alpha_t, sigma_t = self.denoiser.schedule(t)
            tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
            noise = torch.randn_like(x_t)
            with torch.enable_grad():
                x_t = x_t.detach().requires_grad_()
                clean = self.denoiser(x_t, t, **kwargs).mean
                residual = torch.linalg.vector_norm(self.y - self.A(clean))
                gradient = torch.autograd.grad(residual, x_t)[0]
            mean = alpha_s * clean
            mean = mean + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * clean)
            proposal = mean + sigma_s * torch.sqrt(tau) * noise
            amplitude = (mean - (alpha_s / alpha_t) * x_t).norm()
            norm = gradient.norm()
            direction = gradient / torch.where(norm > 0, norm, torch.ones_like(norm))
            return proposal - (beta * amplitude) * direction

    return RelativeAmplitudeSampler
