"""Summarize matched inpainting experiments on the fixed validation panel."""
import argparse
import json
from pathlib import Path


MEAN = "residual_predictive_mean"
SINGLE = "residual_single_draw"
EDGE = "boundary_gradient_mae_k_per_grid_cell"
DEEP = "interior_gt_32"


def rmse(method):
    return method["metrics"]["rmse_k"]


def comparison(candidate, reference):
    if candidate["per_case"].keys() != reference["per_case"].keys():
        raise ValueError("Methods must cover the same cases")
    return {
        "rmse_delta_k": rmse(candidate) - rmse(reference),
        "boundary_gradient_mae_delta": candidate["metrics"][EDGE] - reference["metrics"][EDGE],
        "deep_hole_rmse_delta_k": (
            candidate["distance_layers"][DEEP]["rmse_k"]
            - reference["distance_layers"][DEEP]["rmse_k"]
        ),
        "cases_with_lower_rmse": sum(
            candidate["per_case"][case]["rmse_k"] < baseline["rmse_k"]
            for case, baseline in reference["per_case"].items()
        ),
        "cases": len(reference["per_case"]),
    }


def improves(delta):
    return (delta["rmse_delta_k"] < 0
            and delta["cases_with_lower_rmse"] > delta["cases"] / 2
            and delta["boundary_gradient_mae_delta"] <= 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    reports = {}
    for variant in ("conv", "transformer"):
        for epoch in (10, 20):
            path = directory / f"{variant}-epoch-{epoch:04d}-val45.json"
            reports[variant, epoch] = json.loads(path.read_text())

    pairings = []
    for epoch in (10, 20):
        conv, transformer = (reports[variant, epoch] for variant in ("conv", "transformer"))
        if any(conv[key] != transformer[key] for key in ("samples", "split", "sampling", "seed")):
            raise ValueError("Unmatched validation reports")
        if conv["split_name"] != "val" or len(conv["samples"]) != 45:
            raise ValueError("Expected the 45-frame validation panel")
        pairings.append({
            "epoch": epoch,
            "rmse_k": {
                "unet": rmse(conv["unet"]),
                "conv_single": rmse(conv[SINGLE]),
                "transformer_single": rmse(transformer[SINGLE]),
                "conv_mean": rmse(conv[MEAN]),
                "transformer_mean": rmse(transformer[MEAN]),
            },
            "conv_mean_vs_unet": comparison(conv[MEAN], conv["unet"]),
            "transformer_mean_vs_conv": comparison(transformer[MEAN], conv[MEAN]),
            "transformer_mean_vs_unet": comparison(transformer[MEAN], conv["unet"]),
        })
    hybrid_candidates = [
        row for row in pairings
        if improves(row["conv_mean_vs_unet"])
        and improves(row["transformer_mean_vs_conv"])
        and row["transformer_mean_vs_conv"]["deep_hole_rmse_delta_k"] < 0
    ]
    if hybrid_candidates:
        winner = min(hybrid_candidates, key=lambda row: row["rmse_k"]["transformer_mean"])
        decision = {"next": "confirm_transformer_on_full_validation", "epoch": winner["epoch"]}
    else:
        decision = {"next": "do_not_deepen_transformer_yet"}
    result = json.dumps({"pairings": pairings, "provisional_decision": decision},
                        indent=2, allow_nan=False) + "\n"
    (directory / "validation-comparison.json").write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
