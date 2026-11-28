"""Compose temperature reconstructions and optional field postprocessing."""
import argparse
import json
from pathlib import Path

import numpy as np

from dps_extensions.block_temperature_projection import project_means
from dps_extensions.geometry import partition_edges

ROOT = Path(__file__).resolve().parent
RECIPE = ROOT / "reconstruction_recipe.json"


def load_conditioned_model(device):
    from dps_extensions.coarse_temperature_condition_branch import load_condition
    from flame_reconstruction.runtime import asset, load_model

    path = ROOT / "dps_extensions/coarse_temperature_condition_branch/model_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    minimum, maximum = config["normalization_K"]
    identity = {"ae_sha256": config["ae"]["source_sha256"],
                "prior_sha256": config["prior"]["source_sha256"],
                "min": minimum, "max": maximum}
    model = load_model(path, device)
    return model, load_condition(model, asset(config, "adapter", path.parent), identity)


def postprocess(members, observation, recipe):
    from dps_extensions.temperature_ensemble_mean import ensemble_mean

    mean = ensemble_mean(members)
    if recipe == "mean":
        return mean
    edges = partition_edges(mean.shape, observation.shape)
    aligned = project_means(mean, observation, *edges)
    if recipe == "project":
        return aligned
    from dps_extensions.interpolated_temperature_baseline import interpolate
    baseline = project_means(interpolate(observation, edges), observation, *edges)
    if recipe == "uniform":
        from dps_extensions.uniform_detail_shrinkage import shrink_details
        return project_means(shrink_details(aligned, baseline), observation, *edges)
    from dps_extensions.multiscale_detail_shrinkage import shrink_details
    return shrink_details(aligned, baseline, observation, edges, project_means)


def main():
    config = json.loads(RECIPE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampler", choices=("native", "relative", "condition-branch"), default="native")
    parser.add_argument("--steps", type=int, default=config["fixed_dps"]["steps"])
    parser.add_argument("--zeta", type=float, help="Fixed guidance strength for native or condition-branch")
    parser.add_argument("--beta", type=float, help="Relative guidance multiplier for the relative sampler")
    parser.add_argument("--postprocess", choices=("mean", "project", "uniform", "multiscale", "interpolation"),
                        default="project")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seeds", nargs="+", type=int)
    selection.add_argument("--l5-frame", choices=config["l5_formal_frame_seeds"])
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output path must be new.")
    if args.postprocess == "interpolation":
        from dps_extensions.interpolated_temperature_baseline import interpolate
        observation = np.load(args.observation, allow_pickle=False)
        if observation.shape != (16, 21) or observation.dtype.kind not in "fiu":
            parser.error("Expected 16x21 real block-mean temperatures in K.")
        edges = partition_edges((192, 256), observation.shape)
        field = project_means(interpolate(observation, edges), observation, *edges)
    else:
        from flame_reconstruction.inference import read_observation, reconstruct
        from flame_reconstruction.runtime import load_model, resolve_device
        if args.sampler == "relative" and args.zeta is not None:
            parser.error("Use --beta for relative guidance.")
        if args.beta is not None and args.sampler != "relative":
            parser.error("--beta requires the relative sampler.")
        seeds = args.seeds or config["default_seeds"]
        if args.l5_frame is not None:
            seeds = config["l5_formal_frame_seeds"][args.l5_frame]
        if len(seeds) != len(set(seeds)):
            parser.error("Each ensemble member must have a different seed.")

        observation = read_observation(args.observation)
        device = resolve_device(args.device)
        sampler_factory = condition = None
        if args.sampler == "condition-branch":
            model, condition = load_conditioned_model(device)
        else:
            model = load_model(device=device)
        if args.sampler == "relative":
            from azula.guidance.dps import DPSSampler
            from dps_extensions.relative_amplitude_guidance import relative_sampler
            params = ROOT / "dps_extensions/relative_amplitude_guidance/parameters.json"
            calibrated = json.loads(params.read_text(encoding="utf-8"))["beta"]
            sampler_factory = relative_sampler(DPSSampler, calibrated if args.beta is None else args.beta)
        zeta = config["fixed_dps"]["zeta"] if args.zeta is None else args.zeta
        members = [reconstruct(model, observation, steps=args.steps, zeta=zeta, seed=seed,
                               sampler_factory=sampler_factory, condition=condition) for seed in seeds]
        field = postprocess(members, observation, args.postprocess)

    with args.out.open("xb") as stream:
        np.save(stream, field, allow_pickle=False)


if __name__ == "__main__":
    main()
