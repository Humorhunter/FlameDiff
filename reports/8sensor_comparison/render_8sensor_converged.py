#!/usr/bin/env python3
"""Render fair 8-sensor-only comparison figures:
Converged Original Conditional FlameDiff (ep54) vs. Ours (e59 Unconditional + MMPS).

Usage:
    # Re-render fig6 from committed JSON files in ./data:
    python render_8sensor_converged.py

    # Re-render both fig5 and fig6 when local .npz field dumps are available:
    python render_8sensor_converged.py --fields-dir /path/to/orig-cond-8sensor-20261124
"""
from __future__ import annotations

import argparse
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
DEFAULT_OUT_DIR = HERE / "figures"

SENSOR_COORDS_XY = [
    (20, 95),
    (70, 95),
    (110, 80),
    (110, 110),
    (160, 75),
    (160, 120),
    (200, 75),
    (200, 120),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 220,
})


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_fig5(data_dir: pathlib.Path, fields_dir: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    orig_npz = np.load(fields_dir / "orig_cond_test9_st20_fields.npz")
    ours_npz = np.load(fields_dir / "ours_e59_test9_fields.npz")
    orig_meta = {f["index"]: f for f in load_json(data_dir / "orig_cond_test9_st20.json")["per_frame"]}
    ours_meta = {f["index"]: f for f in load_json(data_dir / "ours_e59_test9_mmps.json")["per_frame"]}

    selected = [
        ("T1030", 75),
        ("T1050", 675),
        ("T1070", 1125),
        ("T1080", 1275),
    ]

    fig = plt.figure(figsize=(16.8, 11.8))
    gs = GridSpec(
        nrows=4,
        ncols=5,
        hspace=0.25,
        wspace=0.07,
        left=0.055,
        right=0.985,
        top=0.865,
        bottom=0.095,
    )

    col_titles = [
        "Ground Truth + 8 Sensors",
        "Converged Orig FlameDiff\n(Cond ep54, 20 steps)",
        "Ours: e59 Uncond + MMPS\n(20 steps, 10 guided)",
        "Orig |Pred - GT| (K)",
        "Ours |Pred - GT| (K)",
    ]

    vmin_t, vmax_t = 300.0, 1600.0
    vmin_e, vmax_e = 0.0, 350.0
    im_temp = None
    im_err = None

    sx = [p[0] for p in SENSOR_COORDS_XY]
    sy = [p[1] for p in SENSOR_COORDS_XY]

    for r, (case_name, idx) in enumerate(selected):
        gt = ours_npz[f"{idx}_truth"]
        pred_orig = orig_npz[f"{idx}_raw"]
        pred_ours = ours_npz[f"{idx}_raw"]
        err_orig = np.abs(pred_orig - gt)
        err_ours = np.abs(pred_ours - gt)

        mo = orig_meta[idx]
        mu = ours_meta[idx]
        rmse_o = mo["raw_metrics"]["rmse_k"]
        obs_o = mo["raw_metrics"]["observation_rmse_k"]
        ssim_o = mo["raw_metrics"]["ssim"]

        rmse_u = mu["raw_metrics"]["rmse_k"]
        obs_u = mu["raw_metrics"]["observation_rmse_k"]
        ssim_u = mu["raw_metrics"]["ssim"]
        delta_rmse = rmse_u - rmse_o

        panels = [
            (gt, "inferno", vmin_t, vmax_t, True, f"Peak {gt.max():.0f} K"),
            (pred_orig, "inferno", vmin_t, vmax_t, False, f"RMSE {rmse_o:.1f} K | 8-Pt {obs_o:.1f} K"),
            (pred_ours, "inferno", vmin_t, vmax_t, False, f"RMSE {rmse_u:.1f} K ({delta_rmse:+.1f}) | 8-Pt {obs_u:.1f} K"),
            (err_orig, "magma", vmin_e, vmax_e, False, f"MAE {mo['raw_metrics']['mae_k']:.1f} K | SSIM {ssim_o:.3f}"),
            (err_ours, "magma", vmin_e, vmax_e, False, f"MAE {mu['raw_metrics']['mae_k']:.1f} K | SSIM {ssim_u:.3f}"),
        ]

        for c, (img, cmap, vmin, vmax, draw_sensors, sub_txt) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
            if c < 3:
                im_temp = im
            else:
                im_err = im
            if draw_sensors:
                ax.scatter(
                    sx, sy, s=28, facecolors="none", edgecolors="#00F5FF", linewidths=1.35, zorder=5
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_edgecolor("#444444")

            if r == 0:
                ax.set_title(f"{col_titles[c]}\n{sub_txt}", fontsize=9.6, fontweight="bold" if c in (1, 2) else "normal", pad=6)
            else:
                ax.set_title(sub_txt, fontsize=9.2, pad=4)

            if c == 0:
                ax.set_ylabel(f"{case_name}\n(idx={idx})", fontsize=10.5, fontweight="bold", labelpad=8)

    cax_t = fig.add_axes([0.075, 0.048, 0.505, 0.017])
    cb_t = fig.colorbar(im_temp, cax=cax_t, orientation="horizontal")
    cb_t.set_label("Temperature (K)  [300 – 1600 K, shared across GT & Reconstructions]", fontsize=9.5)

    cax_e = fig.add_axes([0.645, 0.048, 0.320, 0.017])
    cb_e = fig.colorbar(im_err, cax=cax_e, orientation="horizontal")
    cb_e.set_label("Absolute Error |Pred - GT| (K)  [0 – 350 K, shared]", fontsize=9.5)

    fig.suptitle(
        "8-Sensor Flame Reconstruction: Converged Original FlameDiff (ep54) vs. Ours (e59 Unconditional + MMPS)",
        fontsize=13.5,
        fontweight="bold",
        y=0.968,
    )
    fig.text(
        0.5,
        0.935,
        "Fair controlled benchmark on held-out test set: identical leak-free split, same AE, same ViTDecoder(256, 8L), 60-epoch training budget (raw decoded output before sensor overwrite)",
        ha="center",
        fontsize=9.6,
        color="#333333",
    )

    out_path = out_dir / "fig5_orig_flamediff_vs_ours_8sensor.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def render_fig6(data_dir: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    orig_t20 = load_json(data_dir / "orig_cond_test9_st20.json")
    orig_t50 = load_json(data_dir / "orig_cond_test9_st50.json")
    orig_t1k = load_json(data_dir / "orig_cond_test9_st1000.json")
    ours_t20 = load_json(data_dir / "ours_e59_test9_mmps.json")

    orig_v20 = load_json(data_dir / "orig_cond_val45_st20.json")
    orig_v50 = load_json(data_dir / "orig_cond_val45_st50.json")
    orig_v200 = load_json(data_dir / "orig_cond_val45_st200.json")
    ours_v20 = load_json(data_dir / "sensors_val45_e59.json")

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.6))
    plt.subplots_adjust(left=0.05, right=0.985, top=0.82, bottom=0.14, wspace=0.26)

    c_orig20 = "#4C78A8"
    c_orig1k = "#9ECAE9"
    c_ours = "#E45756"
    c_orig50 = "#72B7B2"

    # Panel A: Per-case Test RMSE
    ax = axes[0]
    cases = [f["case"] for f in orig_t20["per_frame"]]
    r_o20 = [f["metrics"]["rmse_k"] for f in orig_t20["per_frame"]]
    r_o1k = [f["metrics"]["rmse_k"] for f in orig_t1k["per_frame"]]
    r_ours = [f["metrics"]["rmse_k"] for f in ours_t20["per_frame"]]

    x = np.arange(len(cases))
    w = 0.26
    ax.bar(x - w, r_o20, width=w, color=c_orig20, label=f"Orig Cond 20-step (mean {np.mean(r_o20):.1f} K)")
    ax.bar(x, r_o1k, width=w, color=c_orig1k, edgecolor="#4C78A8", linewidth=0.6, label=f"Orig Cond 1000-step (mean {np.mean(r_o1k):.1f} K)")
    ax.bar(x + w, r_ours, width=w, color=c_ours, label=f"Ours e59 MMPS 20-step (mean {np.mean(r_ours):.1f} K)")

    ax.axhline(np.mean(r_o20), color=c_orig20, linestyle="--", linewidth=1.1, alpha=0.8)
    ax.axhline(np.mean(r_ours), color=c_ours, linestyle="--", linewidth=1.2, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=32, ha="right")
    ax.set_ylabel("Full-Field RMSE (K)  [lower is better]")
    ax.set_ylim(0, 224)
    ax.set_title("(a) Per-Case Held-Out Test RMSE (9 Cases)", fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=8.3)

    # Panel B: Quality vs. Speed Pareto (V100S)
    ax = axes[1]
    t9_times = [
        orig_t20["mean_seconds_per_frame"],
        orig_t50["mean_seconds_per_frame"],
        orig_t1k["mean_seconds_per_frame"],
    ]
    t9_rmses = [
        orig_t20["summary_metrics"]["rmse_k"],
        orig_t50["summary_metrics"]["rmse_k"],
        orig_t1k["summary_metrics"]["rmse_k"],
    ]
    v45_times = [
        orig_v20["mean_seconds_per_frame"],
        orig_v50["mean_seconds_per_frame"],
        orig_v200["mean_seconds_per_frame"],
    ]
    v45_rmses = [
        orig_v20["summary_metrics"]["rmse_k"],
        orig_v50["summary_metrics"]["rmse_k"],
        orig_v200["summary_metrics"]["rmse_k"],
    ]

    ax.plot(t9_times, t9_rmses, "-o", color=c_orig20, linewidth=2.0, markersize=7, label="Orig Cond (test9: 20 / 50 / 1000 steps)")
    ax.plot(v45_times, v45_rmses, "--s", color="#54A24B", linewidth=1.8, markersize=6.5, label="Orig Cond (val45: 20 / 50 / 200 steps)")

    ours_t9_t = ours_t20["mean_seconds_per_frame"]
    ours_t9_r = ours_t20["summary_metrics"]["rmse_k"]
    ours_v45_t = ours_v20["mean_seconds_per_frame"]
    ours_v45_r = ours_v20["summary_metrics"]["rmse_k"]

    ax.scatter([ours_t9_t], [ours_t9_r], marker="*", s=240, color=c_ours, edgecolor="black", linewidth=0.7, zorder=6, label="Ours e59 MMPS 20-step (test9)")
    ax.scatter([ours_v45_t], [ours_v45_r], marker="D", s=85, color="#F58518", edgecolor="black", linewidth=0.7, zorder=6, label="Ours e59 MMPS 20-step (val45)")

    ax.annotate("Orig-20\n0.093s, 105.0K", (t9_times[0], t9_rmses[0]), xytext=(-8, 14), textcoords="offset points", fontsize=8.0, color="#2B4C7E")
    ax.annotate("Orig-50\n0.23s, 107.9K", (t9_times[1], t9_rmses[1]), xytext=(8, 10), textcoords="offset points", fontsize=8.0, color="#2B4C7E")
    ax.annotate("Orig-1000 (default)\n4.61s, 118.0K", (t9_times[2], t9_rmses[2]), xytext=(-68, -28), textcoords="offset points", fontsize=8.1, color="#2B4C7E")

    ax.annotate("Orig-20 (val45)\n0.095s, 95.1K", (v45_times[0], v45_rmses[0]), xytext=(10, -22), textcoords="offset points", fontsize=8.0, color="#2D6A2E")
    ax.annotate("Orig-200 (val45)\n0.91s, 106.6K", (v45_times[2], v45_rmses[2]), xytext=(-22, -26), textcoords="offset points", fontsize=8.0, color="#2D6A2E")

    ax.annotate("Ours (test9)\n1.37s, 92.8K", (ours_t9_t, ours_t9_r), xytext=(12, 2), textcoords="offset points", fontsize=8.4, fontweight="bold", color="#B22222")
    ax.annotate("Ours (val45)\n1.48s, 86.8K", (ours_v45_t, ours_v45_r), xytext=(11, -18), textcoords="offset points", fontsize=8.4, fontweight="bold", color="#B85D00")

    ax.set_xscale("log")
    ax.set_xlim(0.06, 11.5)
    ax.set_ylim(78, 131)
    ax.set_xlabel("Inference Time per Frame on V100S (s, log scale)")
    ax.set_ylabel("Full-Field RMSE (K)  [lower is better]")
    ax.set_title("(b) Accuracy vs. Inference Efficiency (V100S)", fontweight="bold", pad=8)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.9)

    # Panel C: Raw 8-Sensor Observation Consistency
    ax = axes[2]
    groups = ["Val 45-Frame Panel", "Test 9-Frame Panel"]
    gx = np.arange(len(groups))
    bw = 0.22

    obs_orig20 = [
        orig_v20["summary_raw_metrics"]["observation_rmse_k"],
        orig_t20["summary_raw_metrics"]["observation_rmse_k"],
    ]
    obs_orig50 = [
        orig_v50["summary_raw_metrics"]["observation_rmse_k"],
        orig_t50["summary_raw_metrics"]["observation_rmse_k"],
    ]
    obs_orig_hi = [
        orig_v200["summary_raw_metrics"]["observation_rmse_k"],
        orig_t1k["summary_raw_metrics"]["observation_rmse_k"],
    ]
    obs_ours = [
        ours_v20["summary_raw_metrics"]["observation_rmse_k"],
        ours_t20["summary_raw_metrics"]["observation_rmse_k"],
    ]

    b1 = ax.bar(gx - 1.5 * bw, obs_orig20, width=bw, color=c_orig20, label="Orig Cond 20-step")
    b2 = ax.bar(gx - 0.5 * bw, obs_orig50, width=bw, color=c_orig50, label="Orig Cond 50-step")
    b3 = ax.bar(gx + 0.5 * bw, obs_orig_hi, width=bw, color=c_orig1k, edgecolor="#4C78A8", linewidth=0.6, label="Orig Cond 200 / 1000-step")
    b4 = ax.bar(gx + 1.5 * bw, obs_ours, width=bw, color=c_ours, label="Ours e59 MMPS 20-step")

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 3.5,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold" if bars is b4 else "normal",
            )

    bbox_kw = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#E45756", alpha=0.92, lw=0.7)
    ax.annotate(
        "2.57x lower\n(45/45 won)",
        xy=(0 + 1.5 * bw, obs_ours[0] + 15),
        xytext=(0 + 1.5 * bw, 116),
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="#B22222",
        bbox=bbox_kw,
        arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.1),
    )
    ax.annotate(
        "2.40x–2.74x lower\n(9/9 won)",
        xy=(1 + 1.5 * bw, obs_ours[1] + 15),
        xytext=(1 + 1.48 * bw, 134),
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="#B22222",
        bbox=bbox_kw,
        arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.1),
    )

    ax.set_xticks(gx)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Raw 8-Sensor Observation RMSE (K)  [lower is better]")
    ax.set_xlim(-0.55, 1.65)
    ax.set_ylim(0, 278)
    ax.set_title("(c) Physical 8-Sensor Consistency (Before Overwrite)", fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=8.0)

    fig.suptitle(
        "8-Sensor Reconstruction Quality, Sensor Consistency & Efficiency: Converged Original FlameDiff vs. Ours",
        fontsize=13.2,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.905,
        "Controlled single-GPU V100S benchmark: Original Conditional FlameDiff (best ep54, val_loss=0.00721) vs. Ours Unconditional Prior (e59, val_loss=0.00884) + Latent MMPS",
        ha="center",
        fontsize=9.5,
        color="#333333",
    )

    out_path = out_dir / "fig6_8sensor_quality_efficiency.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 8-sensor converged comparison figures.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fields-dir", type=pathlib.Path, default=None, help="Directory with *_fields.npz dumps for fig5.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p6 = render_fig6(args.data_dir, args.out_dir)
    print("Saved:", p6)

    if args.fields_dir is not None:
        p5 = render_fig5(args.data_dir, args.fields_dir, args.out_dir)
        print("Saved:", p5)


if __name__ == "__main__":
    main()
