import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from hydra import compose, initialize_config_dir

from dataset import FlameDiffDataset
from training.FlameAgent import FlameAgent
from utils import visualize

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_epoch(path: str) -> int:
    return int(os.path.basename(path).split("epoch=")[1].split("-")[0])


def find_latest_ckpt(exp_root: str) -> str:
    pattern = os.path.join(exp_root, "**", "checkpoints", "flame-*.ckpt")
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No flame-*.ckpt found under {exp_root}")
    return max(ckpts, key=parse_epoch)


def main():
    parser = argparse.ArgumentParser(description="Generate flame fields and compare with the ground truth.")
    parser.add_argument("--ckpt", default=None, help="Path to a flame-*.ckpt. Defaults to the latest under exp/<exp_name>.")
    parser.add_argument("--num", type=int, default=4, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=os.path.join(REPO_ROOT, "inference_output"))
    args = parser.parse_args()

    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "config"), version_base=None):
        cfg = compose(config_name="default_training")

    exp_root = os.path.join(REPO_ROOT, "exp", str(cfg.exp_name))
    ckpt_path = args.ckpt or find_latest_ckpt(exp_root)
    print(f"Checkpoint: {ckpt_path}")

    os.makedirs(args.outdir, exist_ok=True)

    dataset = FlameDiffDataset(cache_path=cfg.cache_path)
    max_v, min_v = dataset.max_min()

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num]
    batch = torch.stack([dataset[int(i)] for i in indices])
    print(f"Sampling indices: {indices.tolist()}")

    agent = FlameAgent(
        endecoder_ckpt=cfg.endecoder_ckpt,
        sensors=cfg.sensors,
        ckpt_path=ckpt_path,
    )
    agent.max_v = max_v
    agent.min_v = min_v

    generated = agent.generate(batch)
    original = batch[:, 0].detach().cpu().numpy()
    generated = generated[:, 0].detach().cpu().numpy()

    for i in range(args.num):
        fig = visualize(original[i], generated[i])
        out_path = os.path.join(args.outdir, f"compare_{i:02d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"Done. Results in {args.outdir}")


if __name__ == "__main__":
    main()
