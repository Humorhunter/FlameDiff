"""Reconstruct a single temperature field using native fixed-strength DPS."""
import argparse

import numpy as np

from .native_dps.posterior import PartitionMeanDownsample, sample_observations
from .runtime import DEFAULT_MODEL_CONFIG, load_model, resolve_device

OBSERVATION_SHAPE = (16, 21)


def read_observation(path):
    field = np.load(path, allow_pickle=False)
    if field.shape != OBSERVATION_SHAPE or field.dtype.kind not in "fiu" or not np.isfinite(field).all():
        raise ValueError(f"Expected a finite {OBSERVATION_SHAPE} block-mean array in K.")
    return field


def reconstruct(model, observation, *, steps=1000, zeta=1.0, seed=0,
                sampler_factory=None, condition=None):
    if np.asarray(observation).shape != OBSERVATION_SHAPE:
        raise ValueError(f"Expected observation shape {OBSERVATION_SHAPE}.")
    operator = PartitionMeanDownsample(low_shape=OBSERVATION_SHAPE)
    result = sample_observations(model, np.asarray(observation).reshape(1, -1), operator,
                                 steps, zeta, seed, sampler_factory=sampler_factory,
                                 condition=condition)
    return result[0, 0].cpu().numpy()


def save_field(path, field):
    with open(path, "xb") as stream:
        np.save(stream, field, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, help="16x21 block-mean .npy in K")
    parser.add_argument("--out", required=True, help="New output .npy path")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    observation = read_observation(args.observation)
    model = load_model(args.model_config, resolve_device(args.device))
    field = reconstruct(model, observation, steps=args.steps, zeta=args.zeta, seed=args.seed)
    save_field(args.out, field)


if __name__ == "__main__":
    main()
