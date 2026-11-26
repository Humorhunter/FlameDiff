"""Train the AE and then the latent DDPM epsilon prior from external fields."""
import argparse
from itertools import islice
import json
from pathlib import Path
import shutil

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from .model.EnDeCoder import AutoEnDecoder
from .runtime import FlameModel, resolve_device, sha256, temperature_range
from .thermal_data import TemperatureFrames


def train_ae(batches, minimum, maximum, device):
    ae = AutoEnDecoder().to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4)
    for frames in batches:
        normalized = (frames.to(device) - minimum) / (maximum - minimum)
        update(mse_loss(ae(normalized), normalized), ae, optimizer)
    return ae.state_dict()


def train_prior(batches, ae_path, minimum, maximum, device):
    model = FlameModel(minimum, maximum).to(device)
    model.autoendecoder.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
    model.autoendecoder.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.vitdecoder.parameters(), lr=1e-5)
    for frames in batches:
        with torch.no_grad():
            latent = model.autoendecoder.encoder(model.normalize(frames.to(device)))
            noise = torch.randn_like(latent)
            times = torch.randint(model.scheduler.config.num_train_timesteps,
                                  (len(latent), 1), device=device)
            noisy_latent = model.scheduler.add_noise(latent, noise, times)
        predicted = model.vitdecoder(noisy_latent, times.float())
        update(mse_loss(predicted, noise), model.vitdecoder, optimizer)
    return model.vitdecoder.state_dict()


def update(loss, network, optimizer):
    if not torch.isfinite(loss):
        raise ArithmeticError("Nonfinite training loss.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()


def repeat_batches(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ae", "prior"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    parser.add_argument("--ae", type=Path, help="AE weights for prior training")
    parser.add_argument("--min-k", required=True, type=float)
    parser.add_argument("--max-k", required=True, type=float)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (args.stage == "prior") != (args.ae is not None):
        parser.error("Pass --ae for prior training only.")
    if args.steps < 1 or args.batch_size < 1:
        parser.error("Expected positive steps and batch size.")
    minimum, maximum = temperature_range(args.min_k, args.max_k)
    if args.ae is not None:
        ae_metadata = args.ae.with_name("training_config.json")
        if ae_metadata.is_file():
            recorded = json.loads(ae_metadata.read_text(encoding="utf-8"))["normalization_K"]
            if recorded != [minimum, maximum]:
                parser.error("Prior normalization must match the trained AE.")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    loader = DataLoader(TemperatureFrames(args.data), batch_size=args.batch_size, shuffle=True)
    batches = islice(repeat_batches(loader), args.steps)
    args.out.mkdir(exist_ok=False)
    if args.stage == "ae":
        state = train_ae(batches, minimum, maximum, device)
        torch.save(state, args.out / "autoencoder.pt")
        metadata = {"normalization_K": [minimum, maximum], "training_steps": args.steps}
        (args.out / "training_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        state = train_prior(batches, args.ae, minimum, maximum, device)
        shutil.copyfile(args.ae, args.out / "autoencoder.pt")
        torch.save(state, args.out / "diffusion_prior.pt")
        config = {name: {"file": file, "sha256": sha256(args.out / file)}
                  for name, file in (("ae", "autoencoder.pt"), ("prior", "diffusion_prior.pt"))}
        config["normalization_K"] = [minimum, maximum]
        config["prior"]["training_steps"] = args.steps
        (args.out / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Completed {args.stage} training: {args.steps} updates")


if __name__ == "__main__":
    main()
