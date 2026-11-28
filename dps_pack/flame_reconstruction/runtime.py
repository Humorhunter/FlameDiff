"""Load the frozen temperature AE and latent DDPM epsilon network."""
import hashlib
import json
import math
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from torch import nn

from .model.EnDeCoder import AutoEnDecoder
from .model.ViTDecoder import ViTDecoder

DEFAULT_MODEL_CONFIG = Path(__file__).with_name("model_config.json")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset(config, name, directory):
    record = config[name]
    path = directory / record["file"]
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Weight SHA-256 mismatch: {path}")
    return path


def make_scheduler():
    return DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="epsilon", clip_sample=True,
        clip_sample_range=1.0, thresholding=False,
    )


def temperature_range(minimum, maximum):
    if not (math.isfinite(minimum) and math.isfinite(maximum) and minimum < maximum):
        raise ValueError("Expected finite minimum and maximum temperatures in increasing order.")
    return minimum, maximum


class FlameModel(nn.Module):
    def __init__(self, minimum, maximum):
        super().__init__()
        self.min_v, self.max_v = temperature_range(minimum, maximum)
        self.autoendecoder = AutoEnDecoder()
        self.vitdecoder = ViTDecoder()
        self.scheduler = make_scheduler()

    def normalize(self, temperatures):
        return (temperatures - self.min_v) / (self.max_v - self.min_v)


def load_model(config_path=DEFAULT_MODEL_CONFIG, device="cpu"):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    directory = config_path.parent
    ae_path = asset(config, "ae", directory)
    prior_path = asset(config, "prior", directory)
    model = FlameModel(*config["normalization_K"])
    model.autoendecoder.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
    model.vitdecoder.load_state_dict(torch.load(prior_path, map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)


def resolve_device(device):
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
