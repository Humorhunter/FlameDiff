"""Condition the historical denoiser on a trained coarse-temperature branch."""
from functools import partial

import torch
from torch.nn.functional import mse_loss

from .adapter import LRConditionBranch


def load_condition(model, path, expected_base):
    """Bind a historical A600 checkpoint to an already verified frozen model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["base"] != expected_base or checkpoint["steps"] != 600:
        raise ValueError("The A600 branch requires its historical AE, prior and normalization.")
    branch = LRConditionBranch(model.vitdecoder.d_model // 2)
    branch.load_state_dict(checkpoint["state_dict"])
    device = next(model.parameters()).device
    branch.to(device).eval().requires_grad_(False)

    def condition(target):
        with torch.no_grad():
            features = branch(target.reshape(-1, 1, 16, 21))
        return partial(model.vitdecoder, condition_features=features)

    return condition


def train_branch(model, operator, frames, *, lr=1e-4, seed=42):
    """Fit the branch from Kelvin tensor batches with an explicitly supplied frozen prior."""
    if model.training or any(param.requires_grad for param in model.parameters()):
        raise ValueError("Pass an eval-mode frozen base model.")
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        branch = LRConditionBranch(model.vitdecoder.d_model // 2).to(device)
    optimizer = torch.optim.Adam(branch.parameters(), lr=lr)
    for batch in frames:
        with torch.no_grad():
            normalized = model.normalize(batch.to(device))
            latent = model.autoendecoder.encoder(normalized)
            observations = operator.forward(normalized).reshape(-1, 1, *operator.low_shape)
            noise = torch.randn(latent.shape, device=device, generator=generator)
            times = torch.randint(model.scheduler.config.num_train_timesteps, (len(latent),),
                                  device=device, generator=generator)
            noisy = model.scheduler.add_noise(latent, noise, times)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.vitdecoder(noisy, times[:, None].float(), branch(observations))
        loss = mse_loss(prediction, noise)
        if not torch.isfinite(loss):
            raise ArithmeticError("Nonfinite adapter training loss.")
        loss.backward()
        optimizer.step()
    return branch.eval()
