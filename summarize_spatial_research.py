#!/usr/bin/env python3
"""Summarize the edge-regression and mask-generalization experiments.

The script consumes the benchmark JSON files emitted by
``spatial_edge_regression.py`` and ``spatial_mask_study.py``. It intentionally
keeps the deterministic-edge and mask-diffusion tracks separate: their model
training protocols differ, while the mask-study diffusion is a residual model
on a separately trained shared U-Net.

Example:
    python summarize_spatial_research.py \
        --edge-dir outputs/edge-regression-30ep \
        --mask-dir outputs/mask-folds \
        --output reports/spatial-research/report.md
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import mean


HOT_THRESHOLD_K = 1200.0
EDGE_FILES = {"validation": "validation.json", "test": "test.json"}
MASK_FOLDS = ("within-case", "T1050", "T1080")
MASK_NAMES = ("small", "medium", "large")
METHOD_LABELS = {
    "unet": "冻结 U-Net",
    "edge_regression": "U-Net + 确定性边界修正",
    "diffusion_taper_8_single_draw": "残差扩散单次采样",
    "diffusion_taper_8_predictive_mean": "残差扩散多次均值",
    "shared_unet": "多尺寸共享 U-Net",
    "residual_diffusion": "U-Net + 残差扩散",
    "harmonic": "调和插值",
    "dps": "DPS",
}
METRIC_LABELS = {
    "rmse_k": "RMSE (K)",
    "mae_k": "MAE (K)",
    "ssim": "SSIM",
    "boundary_gradient_mae_k_per_grid_cell": "边界梯度误差 (K/格)",
    "gradient_mae_k_per_grid_cell": "梯度误差 (K/格)",
    "observation_rmse_k": "已知区域 RMSE (K)",
}


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"缺少实验结果文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 无法解析：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"期望 JSON 对象：{path}")
    return value


def _require(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise ValueError(f"{context} 缺少字段 {key!r}")
    return mapping[key]


def _sample_key(sample: dict) -> tuple[str, int]:
    try:
        return str(sample["path"]), int(sample["frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"无效的样本标识，应包含 path 和 frame：{sample}") from exc


def _unique_sample_keys(samples: list, context: str) -> set[tuple[str, int]]:
    keys = [_sample_key(sample) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} 中存在重复的 (path, frame) 样本")
    return set(keys)


def _check_threshold(value, context: str):
    if value is None or not math.isclose(float(value), HOT_THRESHOLD_K, abs_tol=1e-8):
        raise ValueError(
            f"{context} 的高温阈值为 {value!r} K；报告要求统一使用 "
            f"{HOT_THRESHOLD_K:g} K，才能比较高温带指标"
        )


def _load_edge_reports(directory: Path) -> dict:
    reports = {}
    for split, filename in EDGE_FILES.items():
        path = directory / filename
        report = _load_json(path)
        actual_split = report.get("split_name")
        expected_split = "val" if split == "validation" else "test"
        if actual_split != expected_split:
            raise ValueError(
                f"{path} 的 split_name={actual_split!r}，应为 {expected_split!r}"
            )
        protocol = _require(report, "hot_band_protocol", str(path))
        _check_threshold(protocol.get("threshold_k"), str(path))
        methods = _require(report, "methods", str(path))
        if "unet" not in methods:
            raise ValueError(f"{path} 缺少 U-Net 基线")
        sample_keys = _unique_sample_keys(
            _require(report, "samples", str(path)), f"{path}:samples"
        )
        method_keys = {}
        case_keys = {}
        for name, method in methods.items():
            rows = _require(method, "per_frame", f"{path}:methods.{name}")
            hot = method.get("hot_band_metrics", {})
            required_hot_fields = {
                "reference_coverage", "detection_coverage_given_reference",
                "mean_dice_nonempty_union", "mean_centroid_displacement_px",
                "dice_valid_frames", "centroid_valid_frames",
            }
            missing = required_hot_fields - set(hot)
            if missing:
                raise ValueError(
                    f"{path} methods.{name} 缺少高温覆盖字段：{', '.join(sorted(missing))}"
                )
            keys = _unique_sample_keys(rows, f"{path}:methods.{name}.per_frame")
            if sample_keys and keys != sample_keys:
                raise ValueError(
                    f"{path} 中 {name} 的逐帧记录与 samples 不匹配 "
                    f"(samples={len(sample_keys)}, rows={len(keys)})"
                )
            method_keys[name] = keys
            if "per_case" in method:
                case_keys[name] = set(method["per_case"])
        if any(keys != method_keys["unet"] for keys in method_keys.values()):
            raise ValueError(f"{path} 的方法未在完全相同的帧上评估")
        if case_keys and any(keys != case_keys.get("unet", set()) for keys in case_keys.values()):
            raise ValueError(f"{path} 的方法未覆盖完全相同的工况")
        reports[split] = report
    return reports


def _load_mask_reports(directory: Path) -> dict:
    reports = {}
    for fold in MASK_FOLDS:
        path = directory / fold / "test.json"
        report = _load_json(path)
        if report.get("status") != "complete":
            raise ValueError(f"{path} 尚未完成：status={report.get('status')!r}")
        if report.get("split_name") != "test":
            raise ValueError(f"{path} 不是 test split")
        expected_case = None if fold == "within-case" else fold
        actual_case = report.get("test_case")
        if actual_case != expected_case:
            raise ValueError(
                f"{path} 的 test_case={actual_case!r}，预期为 {expected_case!r}"
            )
        evaluation = _require(report, "evaluation", str(path))
        _check_threshold(evaluation.get("high_temperature_threshold_k"), str(path))
        samples = _require(evaluation, "samples", f"{path}:evaluation")
        keys = _unique_sample_keys(samples, f"{path}:evaluation.samples")
        models = _require(report, "models", str(path))
        if "shared_unet" not in models:
            raise ValueError(f"{path} 缺少 shared_unet 基线")
        for method_name, mask_records in models.items():
            for mask_name, record in mask_records.items():
                if record.get("frames") != len(keys):
                    raise ValueError(
                        f"{path} {method_name}/{mask_name} 的 frames 与 evaluation.samples 不一致"
                    )
                hot_key = (
                    "predictive_mean_high_temperature_hole_metrics"
                    if method_name == "residual_diffusion"
                    else "high_temperature_hole_metrics"
                )
                hot = record.get(hot_key, {})
                required_hot_fields = {
                    "reference_coverage", "detection_coverage_given_reference",
                    "mean_dice_nonempty_union", "mean_centroid_displacement_px",
                    "dice_valid_frames", "centroid_valid_frames",
                }
                missing = required_hot_fields - set(hot)
                if missing:
                    raise ValueError(
                        f"{path} {method_name}/{mask_name} 缺少新版高温覆盖字段："
                        f"{', '.join(sorted(missing))}；请使用修正 Dice 汇总后的 benchmark JSON"
                    )
        reports[fold] = report
    return reports


def _fmt(value, digits=2, suffix="") -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}{suffix}"


def _fmt_signed(value, digits=2, suffix="") -> str:
    text = _fmt(value, digits, suffix)
    if text == "—":
        return text
    return f"+{text}" if float(value) > 0 else text


def _delta(candidate, baseline):
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def _plot_label(method: str) -> str:
    return {
        "unet": "U-Net",
        "edge_regression": "U-Net + edge",
        "diffusion_taper_8_single_draw": "Diffusion · 1 draw",
        "diffusion_taper_8_predictive_mean": "Diffusion · mean",
    }.get(method, method.replace("_", " "))


def _edge_hot_summary(method: dict) -> dict:
    return method.get("hot_band_metrics", {})


def _mask_hot_summary(record: dict, residual: bool) -> dict:
    key = (
        "predictive_mean_high_temperature_hole_metrics"
        if residual else "high_temperature_hole_metrics"
    )
    summary = record.get(key, {})
    return {
        "mean_centroid_displacement_px": summary.get("mean_centroid_displacement_px"),
        "mean_dice_nonempty_union": summary.get("mean_dice_nonempty_union"),
        "reference_coverage": summary.get("reference_coverage"),
        "detection_coverage_given_reference": summary.get("detection_coverage_given_reference"),
        "centroid_valid_coverage": summary.get("centroid_valid_coverage"),
        "centroid_valid_frames": summary.get("centroid_valid_frames"),
        "dice_valid_frames": summary.get("dice_valid_frames"),
    }


def _percent(value) -> str:
    return _fmt(None if value is None else 100.0 * float(value), 1, "%")


def _edge_rows_table(reports: dict) -> list[str]:
    lines = [
        "| 数据划分 | 方法 | 帧数 | RMSE (K) | ΔRMSE vs U-Net (K) | "
        "边界梯度误差 (K/格) | Δ边界误差 | 真值热区覆盖 | 检出率 | "
        "质心/Dice 有效帧 | 热区质心偏差 (格) | Dice | 秒/帧 | checkpoint epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in ("validation", "test"):
        report = reports[split]
        methods = report["methods"]
        base = methods["unet"].get("metrics", {})
        frames = len(report.get("samples", []))
        for name in methods:
            method = methods[name]
            metrics = method.get("metrics", {})
            hot = _edge_hot_summary(method)
            timing = method.get("timing", {})
            if name == "unet":
                epoch = f"U-Net {report.get('unet_epoch', '—')}"
            elif name == "edge_regression":
                epoch = f"U-Net {report.get('unet_epoch', '—')} + edge {report.get('edge_epoch', '—')}"
            else:
                checkpoint = report.get("diffusion_checkpoint")
                epoch = f"U-Net {report.get('unet_epoch', '—')} + {Path(checkpoint).name if checkpoint else 'diffusion —'}"
            label = _label(name)
            sampling = report.get("diffusion_sampling") or {}
            if name == "diffusion_taper_8_single_draw":
                label += " (1 draw)"
            elif name == "diffusion_taper_8_predictive_mean" and sampling.get("draws_per_frame") is not None:
                label += f" ({sampling['draws_per_frame']} draws)"
            lines.append(
                f"| {split} | {label} | {frames} | "
                f"{_fmt(metrics.get('rmse_k'))} | "
                f"{_fmt(_delta(metrics.get('rmse_k'), base.get('rmse_k')), suffix='')} | "
                f"{_fmt(metrics.get('boundary_gradient_mae_k_per_grid_cell'))} | "
                f"{_fmt(_delta(metrics.get('boundary_gradient_mae_k_per_grid_cell'), base.get('boundary_gradient_mae_k_per_grid_cell')))} | "
                f"{_percent(hot.get('reference_coverage'))} | "
                f"{_percent(hot.get('detection_coverage_given_reference'))} | "
                f"{hot.get('centroid_valid_frames', '—')}/{hot.get('dice_valid_frames', '—')} | "
                f"{_fmt(hot.get('mean_centroid_displacement_px'))} | "
                f"{_fmt(hot.get('mean_dice_nonempty_union'), 3)} | "
                f"{_fmt(timing.get('mean_seconds_per_frame_end_to_end'), 3)} | {epoch} |"
            )
    return lines


def _edge_per_case_table(report: dict) -> list[str]:
    methods = report["methods"]
    baseline = methods["unet"].get("per_case", {})
    candidates = [name for name in methods if name != "unet"]
    lines = ["| 工况 |" + "".join(
        f" {_label(name)} ΔRMSE (K) | {_label(name)} Δ边界误差 |" for name in candidates
    )]
    lines.append("|---|" + "".join("---:|---:|" for _ in candidates))
    for case in sorted(baseline):
        cells = [case]
        for name in candidates:
            case_metrics = methods[name].get("per_case", {}).get(case, {})
            base_metrics = baseline[case]
            cells.extend([
                _fmt_signed(_delta(case_metrics.get("rmse_k"), base_metrics.get("rmse_k"))),
                _fmt_signed(_delta(
                    case_metrics.get("boundary_gradient_mae_k_per_grid_cell"),
                    base_metrics.get("boundary_gradient_mae_k_per_grid_cell"),
                )),
            ])
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _mask_method_record(report: dict, method: str, mask_name: str) -> dict | None:
    return report.get("models", {}).get(method, {}).get(mask_name)


def _mask_metrics(record: dict | None, residual: bool) -> dict:
    if not record:
        return {}
    return record.get("predictive_mean_metrics", {}) if residual else record.get("metrics", {})


def _mask_latency(record: dict | None, residual: bool) -> float | None:
    if not record:
        return None
    if not residual:
        return record.get("mean_inference_seconds_per_frame")
    unet_seconds = record.get("mean_unet_seconds_per_frame")
    residual_seconds = record.get("mean_residual_seconds_per_draw")
    draws = record.get("draws_per_frame")
    if unet_seconds is None or residual_seconds is None or draws is None:
        return None
    return float(unet_seconds) + int(draws) * float(residual_seconds)


def _mask_training_epochs(report: dict) -> str:
    unet_epoch = report.get("training", {}).get("epoch")
    diffusion_epoch = report.get("residual_diffusion", {}).get("epoch")
    return f"{unet_epoch if unet_epoch is not None else '—'} / {diffusion_epoch if diffusion_epoch is not None else '—'}"


def _mask_rows_table(reports: dict) -> list[str]:
    lines = [
        "| 测试设置 | 遮挡尺寸 | 帧数 | 共享 U-Net RMSE | 扩散预测均值 RMSE | ΔRMSE (K) | "
        "Δ边界误差 (K/格) | 真值热区覆盖 | 检出率 U-Net→扩散 | "
        "质心/Dice 有效帧 U-Net→扩散 | Dice U-Net→扩散 | "
        "质心偏差 U-Net→扩散 (格) | 秒/帧 U-Net→扩散（draws） | 训练 epoch U-Net/扩散 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fold in MASK_FOLDS:
        report = reports[fold]
        sizes = report.get("mask_sizes", {})
        samples = report.get("evaluation", {}).get("samples", [])
        for mask_name in MASK_NAMES:
            if mask_name not in sizes:
                continue
            unet = _mask_method_record(report, "shared_unet", mask_name)
            diffusion = _mask_method_record(report, "residual_diffusion", mask_name)
            unet_metrics = _mask_metrics(unet, residual=False)
            diffusion_metrics = _mask_metrics(diffusion, residual=True)
            unet_hot = _mask_hot_summary(unet or {}, residual=False)
            diffusion_hot = _mask_hot_summary(diffusion or {}, residual=True)
            height, width = sizes[mask_name]
            delta_rmse = _delta(diffusion_metrics.get("rmse_k"), unet_metrics.get("rmse_k"))
            delta_boundary = _delta(
                diffusion_metrics.get("boundary_gradient_mae_k_per_grid_cell"),
                unet_metrics.get("boundary_gradient_mae_k_per_grid_cell"),
            )
            lines.append(
                f"| {fold} | {mask_name} ({height}×{width}) | {len(samples)} | "
                f"{_fmt(unet_metrics.get('rmse_k'))} | {_fmt(diffusion_metrics.get('rmse_k'))} | "
                f"{_fmt(delta_rmse, suffix='')} | {_fmt(delta_boundary)} | "
                f"{_percent(unet_hot.get('reference_coverage'))} | "
                f"{_percent(unet_hot.get('detection_coverage_given_reference'))}→{_percent(diffusion_hot.get('detection_coverage_given_reference'))} | "
                f"{unet_hot.get('centroid_valid_frames', '—')}/{unet_hot.get('dice_valid_frames', '—')}→"
                f"{diffusion_hot.get('centroid_valid_frames', '—')}/{diffusion_hot.get('dice_valid_frames', '—')} | "
                f"{_fmt(unet_hot.get('mean_dice_nonempty_union'), 3)}→{_fmt(diffusion_hot.get('mean_dice_nonempty_union'), 3)} | "
                f"{_fmt(unet_hot.get('mean_centroid_displacement_px'))}→{_fmt(diffusion_hot.get('mean_centroid_displacement_px'))} | "
                f"{_fmt(_mask_latency(unet, False), 3)}→{_fmt(_mask_latency(diffusion, True), 3)} "
                f"({diffusion.get('draws_per_frame', '—') if diffusion else '—'} draws) | "
                f"{_mask_training_epochs(report)} |"
            )
    return lines


def _numeric_frame_summary(rows: list[dict]) -> dict:
    keys = (
        "mae_k", "rmse_k", "gradient_mae_k_per_grid_cell",
        "boundary_gradient_mae_k_per_grid_cell", "ssim", "observation_rmse_k",
    )
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        out[key] = mean(values) if values else None
    hot_rows = [row.get("hot_band", {}) for row in rows]
    for target, source in (
        ("mean_centroid_displacement_px", "centroid_displacement_px"),
        ("mean_dice_nonempty_union", "dice"),
    ):
        values = [float(row[source]) for row in hot_rows if row.get(source) is not None]
        out[target] = mean(values) if values else None
    reference_frames = sum(row.get("reference_pixels", 0) > 0 for row in hot_rows)
    detected_frames = sum(
        row.get("reference_pixels", 0) > 0 and row.get("prediction_pixels", 0) > 0
        for row in hot_rows
    )
    centroid_frames = sum(row.get("centroid_displacement_px") is not None for row in hot_rows)
    out["reference_coverage"] = reference_frames / len(rows) if rows else None
    out["detection_coverage_given_reference"] = (
        detected_frames / reference_frames if reference_frames else None
    )
    out["centroid_valid_coverage"] = centroid_frames / len(rows) if rows else None
    out["centroid_valid_frames"] = centroid_frames
    out["dice_valid_frames"] = sum(row.get("dice") is not None for row in hot_rows)
    return out


def _within_case_matched_edge(edge_test: dict, within_report: dict) -> tuple[list[str], dict] | None:
    sample_keys = _unique_sample_keys(
        within_report.get("evaluation", {}).get("samples", []),
        "mask within-case evaluation.samples",
    )
    edge_range = edge_test.get("temperature_range_k")
    mask_range = within_report.get("temperature_range_k")
    if edge_range is None or mask_range is None or len(edge_range) != 2 or len(mask_range) != 2:
        return None
    if any(not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)
           for a, b in zip(edge_range, mask_range)):
        return None
    edge_methods = edge_test.get("methods", {})
    if "unet" not in edge_methods:
        return None
    indices = {}
    for name, method in edge_methods.items():
        index = {_sample_key(row): row for row in method.get("per_frame", [])}
        if not sample_keys.issubset(index):
            return None
        indices[name] = index
    count = len(sample_keys)
    rows = [
        "| 方法/模型 | RMSE (K) | 边界梯度误差 (K/格) | 真值热区覆盖 | 检出率 | "
        "质心/Dice 有效帧 | 热区质心偏差 (格) | Dice |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in edge_methods.items():
        summary = _numeric_frame_summary([indices[name][key] for key in sample_keys])
        rows.append(
            f"| {_label(name)}（边界实验） | {_fmt(summary['rmse_k'])} | "
            f"{_fmt(summary['boundary_gradient_mae_k_per_grid_cell'])} | "
            f"{_percent(summary['reference_coverage'])} | "
            f"{_percent(summary['detection_coverage_given_reference'])} | "
            f"{summary['centroid_valid_frames']}/{summary['dice_valid_frames']} | "
            f"{_fmt(summary['mean_centroid_displacement_px'])} | "
            f"{_fmt(summary['mean_dice_nonempty_union'], 3)} |"
        )
    mask_sizes = within_report.get("mask_sizes", {})
    if tuple(mask_sizes.get("medium", ())) == (96, 128):
        for name, residual in (("shared_unet", False), ("residual_diffusion", True)):
            record = _mask_method_record(within_report, name, "medium")
            if record is None:
                continue
            metrics = _mask_metrics(record, residual)
            hot = _mask_hot_summary(record, residual)
            rows.append(
                f"| {_label(name)}（遮挡研究，medium） | {_fmt(metrics.get('rmse_k'))} | "
                f"{_fmt(metrics.get('boundary_gradient_mae_k_per_grid_cell'))} | "
                f"{_percent(hot.get('reference_coverage'))} | "
                f"{_percent(hot.get('detection_coverage_given_reference'))} | "
                f"{hot.get('centroid_valid_frames', '—')}/{hot.get('dice_valid_frames', '—')} | "
                f"{_fmt(hot.get('mean_centroid_displacement_px'))} | "
                f"{_fmt(hot.get('mean_dice_nonempty_union'), 3)} |"
            )
    return rows, {"frames": count, "sample_keys": sample_keys}


def _edge_conclusions(reports: dict) -> list[str]:
    test = reports["test"]["methods"]
    baseline = test["unet"].get("metrics", {})
    lines = []
    candidates = [name for name in test if name != "unet"]
    if not candidates:
        return ["- 独立测试文件只有 U-Net 基线，当前无法评价候选方法。"]
    for name in candidates:
        metrics = test[name].get("metrics", {})
        delta_rmse = _delta(metrics.get("rmse_k"), baseline.get("rmse_k"))
        delta_edge = _delta(
            metrics.get("boundary_gradient_mae_k_per_grid_cell"),
            baseline.get("boundary_gradient_mae_k_per_grid_cell"),
        )
        quality = (
            "RMSE 与边界梯度误差均下降"
            if delta_rmse is not None and delta_edge is not None and delta_rmse < 0 and delta_edge < 0
            else "RMSE 下降但边界梯度误差上升"
            if delta_rmse is not None and delta_edge is not None and delta_rmse < 0 <= delta_edge
            else "RMSE 上升但边界梯度误差下降"
            if delta_rmse is not None and delta_edge is not None and delta_rmse >= 0 > delta_edge
            else "两项误差均未下降"
        )
        lines.append(
            f"- 测试集上，{_label(name)}相对冻结 U-Net 的 ΔRMSE 为 "
            f"{_fmt(delta_rmse)} K、Δ边界梯度误差为 {_fmt(delta_edge)} K/格；{quality}。"
        )
        base_case = test["unet"].get("per_case", {})
        method_case = test[name].get("per_case", {})
        common_cases = sorted(set(base_case) & set(method_case))
        if common_cases:
            case_rmse_wins = sum(
                method_case[case].get("rmse_k", math.inf) < base_case[case].get("rmse_k", math.inf)
                for case in common_cases
            )
            case_edge_wins = sum(
                method_case[case].get("boundary_gradient_mae_k_per_grid_cell", math.inf)
                < base_case[case].get("boundary_gradient_mae_k_per_grid_cell", math.inf)
                for case in common_cases
            )
            lines.append(
                f"  配对工况中 RMSE 较低 `{case_rmse_wins}/{len(common_cases)}` 个，"
                f"边界梯度误差较低 `{case_edge_wins}/{len(common_cases)}` 个。"
            )
    return lines


def _mask_conclusions(reports: dict) -> list[str]:
    lines = []
    for fold in MASK_FOLDS:
        report = reports[fold]
        changes = []
        for mask_name in MASK_NAMES:
            unet = _mask_method_record(report, "shared_unet", mask_name)
            diffusion = _mask_method_record(report, "residual_diffusion", mask_name)
            if unet is None or diffusion is None:
                continue
            unet_rmse = _mask_metrics(unet, False).get("rmse_k")
            diffusion_rmse = _mask_metrics(diffusion, True).get("rmse_k")
            if unet_rmse is not None and diffusion_rmse is not None:
                changes.append((mask_name, float(diffusion_rmse) - float(unet_rmse)))
        if not changes:
            lines.append(f"- `{fold}` 没有完整的共享 U-Net/残差扩散配对结果。")
            continue
        improved = [name for name, delta in changes if delta < 0]
        worst = max(changes, key=lambda item: item[1])
        best = min(changes, key=lambda item: item[1])
        if improved:
            summary = f"扩散在 {len(improved)}/{len(changes)} 个遮挡尺寸上 RMSE 较低（{', '.join(improved)}）"
        else:
            summary = f"扩散在 {len(changes)} 个遮挡尺寸上 RMSE 都高于共享 U-Net"
        lines.append(
            f"- `{fold}` 测试：{summary}；ΔRMSE 范围 {_fmt(best[1])} 到 {_fmt(worst[1])} K。"
        )
    return lines


def _edge_figure_data(reports: dict) -> dict:
    rows = []
    for split in ("validation", "test"):
        methods = reports[split]["methods"]
        base = methods["unet"].get("metrics", {})
        for name, method in methods.items():
            if name == "unet":
                continue
            metrics = method.get("metrics", {})
            rows.append({
                "split": split,
                "method": _label(name),
                "plot_method": _plot_label(name),
                "rmse_delta": _delta(metrics.get("rmse_k"), base.get("rmse_k")),
                "edge_delta": _delta(
                    metrics.get("boundary_gradient_mae_k_per_grid_cell"),
                    base.get("boundary_gradient_mae_k_per_grid_cell"),
                ),
                "latency": method.get("timing", {}).get("mean_seconds_per_frame_end_to_end"),
                "rmse": metrics.get("rmse_k"),
            })
    return {"rows": rows}


def _mask_figure_data(reports: dict) -> dict:
    rows = []
    for fold in MASK_FOLDS:
        report = reports[fold]
        for mask_name in MASK_NAMES:
            unet = _mask_method_record(report, "shared_unet", mask_name)
            diffusion = _mask_method_record(report, "residual_diffusion", mask_name)
            if not unet:
                continue
            rows.append({
                "fold": fold,
                "mask": mask_name,
                "unet_rmse": _mask_metrics(unet, False).get("rmse_k"),
                "diffusion_rmse": _mask_metrics(diffusion, True).get("rmse_k") if diffusion else None,
                "unet_seconds": _mask_latency(unet, False),
                "diffusion_seconds": _mask_latency(diffusion, True) if diffusion else None,
            })
    return {"rows": rows}


def _write_svg_bars(path: Path, title: str, y_label: str, entries: list[tuple[str, float]], *, zero_line=False):
    """Dependency-free fallback for compact numeric bar charts."""
    entries = [(label, float(value)) for label, value in entries if value is not None and math.isfinite(float(value))]
    if not entries:
        return False
    width, height = max(900, 54 * len(entries)), 500
    left, right, top, bottom = 90, 25, 65, 165
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for _, v in entries]
    lo = min(0.0, min(values)) if zero_line else min(0.0, min(values) * 0.94)
    hi = max(0.0, max(values) * 1.12)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    y = lambda v: top + (hi - v) / (hi - lo) * plot_h
    zero_y = y(0.0)
    bar_w = max(8.0, plot_w / max(1, len(entries)) * 0.55)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial,sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="13">{html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#666" stroke-width="1"/>',
    ]
    for i, (label, value) in enumerate(entries):
        center = left + (i + 0.5) * plot_w / len(entries)
        value_y = y(value)
        rect_y = min(value_y, zero_y)
        rect_h = max(1.0, abs(zero_y - value_y))
        color = "#2a6fbb" if value >= 0 else "#2e8b57"
        parts.append(f'<rect x="{center-bar_w/2:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{center:.1f}" y="{max(top+12,value_y-5):.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11">{value:.2f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+22}" transform="rotate(-38 {center:.1f} {top+plot_h+22})" text-anchor="end" font-family="Arial,sans-serif" font-size="10">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return True


def _make_figures(output_dir: Path, edge_reports: dict, mask_reports: dict) -> list[str]:
    edge_data = _edge_figure_data(edge_reports)["rows"]
    mask_data = _mask_figure_data(mask_reports)["rows"]
    artifacts = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is None:
        edge_rmse = [(f"{r['split']} / {r['plot_method']}", r["rmse_delta"]) for r in edge_data]
        edge_boundary = [(f"{r['split']} / {r['plot_method']}", r["edge_delta"]) for r in edge_data]
        if _write_svg_bars(output_dir / "edge-delta-rmse.svg", "Edge study: RMSE delta vs U-Net", "Δ RMSE (K)", edge_rmse, zero_line=True):
            artifacts.append("edge-delta-rmse.svg")
        if _write_svg_bars(output_dir / "edge-delta-boundary.svg", "Edge study: boundary error delta vs U-Net", "Δ boundary error (K/grid)", edge_boundary, zero_line=True):
            artifacts.append("edge-delta-boundary.svg")
        mask_entries = []
        for row in mask_data:
            for method, key in (("U-Net", "unet_rmse"), ("Diffusion mean", "diffusion_rmse")):
                if row.get(key) is not None:
                    mask_entries.append((f"{row['fold']} / {row['mask']} / {method}", row[key]))
        if _write_svg_bars(output_dir / "mask-rmse.svg", "Mask study: missing-region RMSE", "RMSE (K)", mask_entries):
            artifacts.append("mask-rmse.svg")
        cost_entries = []
        for method, record in edge_reports["test"]["methods"].items():
            latency = record.get("timing", {}).get("mean_seconds_per_frame_end_to_end")
            if latency is not None:
                cost_entries.append((f"edge test / {_plot_label(method)}", latency))
        for row in mask_data:
            if row.get("mask") == "medium":
                for method, key in (("U-Net", "unet_seconds"), ("Diffusion mean", "diffusion_seconds")):
                    if row.get(key) is not None:
                        cost_entries.append((f"{row['fold']} medium / {method}", row[key]))
        if _write_svg_bars(output_dir / "inference-cost.svg", "Measured inference time", "seconds / frame", cost_entries):
            artifacts.append("inference-cost.svg")
        return artifacts

    # Edge ablation: paired metric deltas relative to the same split's U-Net.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for ax, key, title, ylabel in (
        (axes[0], "rmse_delta", "Missing-region RMSE change", "Δ RMSE (K)"),
        (axes[1], "edge_delta", "Boundary-gradient change", "Δ boundary error (K/grid)")
    ):
        labels = [f"{r['split']}\n{r['plot_method']}" for r in edge_data]
        values = [r[key] for r in edge_data]
        colors = ["#4c78a8" if r["split"] == "validation" else "#f58518" for r in edge_data]
        ax.bar(range(len(values)), values, color=colors)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(range(len(labels)), labels, rotation=32, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Deterministic edge correction and diffusion vs U-Net")
    fig.tight_layout()
    fig.savefig(output_dir / "edge-deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts.append("edge-deltas.png")

    # Mask-size response, with separate panels for the within-case and two
    # held-out-condition tests.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    size_index = {name: index for index, name in enumerate(MASK_NAMES)}
    for ax, fold in zip(axes, MASK_FOLDS):
        records = [row for row in mask_data if row["fold"] == fold]
        unet_values = [None] * len(MASK_NAMES)
        diffusion_values = [None] * len(MASK_NAMES)
        for row in records:
            index = size_index[row["mask"]]
            unet_values[index] = row.get("unet_rmse")
            diffusion_values[index] = row.get("diffusion_rmse")
        ax.plot(range(3), unet_values, marker="o", label="Shared U-Net", color="#4c78a8")
        if any(value is not None for value in diffusion_values):
            ax.plot(range(3), diffusion_values, marker="s", label="Residual diffusion mean", color="#e45756")
        ax.set_xticks(range(3), ["small", "medium", "large"])
        ax.set_title(fold)
        ax.set_xlabel("Centered hole size")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Missing-region RMSE (K)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Mask-size study: within-case and held-out conditions")
    fig.tight_layout()
    fig.savefig(output_dir / "mask-size-rmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts.append("mask-size-rmse.png")

    # Timing is intentionally faceted by experiment; the protocols and
    # architectures differ, so this chart is descriptive rather than a single
    # leaderboard across tracks.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    edge_test = [
        (name, record.get("timing", {}).get("mean_seconds_per_frame_end_to_end"))
        for name, record in edge_reports["test"]["methods"].items()
        if record.get("timing", {}).get("mean_seconds_per_frame_end_to_end") is not None
    ]
    axes[0].bar([_plot_label(name) for name, _ in edge_test], [latency for _, latency in edge_test], color="#4c78a8")
    axes[0].set_title("Edge track · full test frame")
    axes[0].set_ylabel("Seconds / frame")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    medium = [row for row in mask_data if row["mask"] == "medium"]
    cost_labels, cost_values, cost_colors = [], [], []
    for row in medium:
        if row.get("unet_seconds") is not None:
            cost_labels.append(f"{row['fold']}\nU-Net")
            cost_values.append(row["unet_seconds"])
            cost_colors.append("#4c78a8")
        if row.get("diffusion_seconds") is not None:
            cost_labels.append(f"{row['fold']}\nDiffusion")
            cost_values.append(row["diffusion_seconds"])
            cost_colors.append("#e45756")
    axes[1].bar(cost_labels, cost_values, color=cost_colors)
    axes[1].set_title("Mask track · medium hole")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Measured inference time (tracks are not directly comparable)")
    fig.tight_layout()
    fig.savefig(output_dir / "inference-cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts.append("inference-cost.png")
    return artifacts


def build_report(edge_reports: dict, mask_reports: dict, output_path: Path) -> list[str]:
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = _make_figures(output_dir, edge_reports, mask_reports)
    matched = _within_case_matched_edge(
        edge_reports["test"], mask_reports["within-case"]
    )
    edge_figure = next((name for name in figures if name.startswith("edge-deltas")), None)
    edge_fallback_figures = [name for name in figures if name.startswith("edge-delta-")]
    mask_figure = next((name for name in figures if name.startswith("mask-size-rmse")), None)
    mask_fallback_figure = next((name for name in figures if name.startswith("mask-rmse")), None)
    cost_figure = next((name for name in figures if name.startswith("inference-cost")), None)

    lines = [
        "# FlameDiff 空间重建实验阶段总结",
        "",
        "## 结论摘要",
        "",
        *_edge_conclusions(edge_reports),
        *_mask_conclusions(mask_reports),
        "- 上述结果按各 JSON 中的样本与指标汇总；它们用于判断当前方向，不能替代多随机种子重复实验。",
        "",
        "## 确定性边界修正与扩散对照",
        "",
        "独立测试集结果见下表。RMSE 与边界梯度误差的差值均以同一划分的冻结 U-Net 为参照；负值表示误差降低。",
        "",
        *_edge_rows_table(edge_reports),
        "",
        "### 独立测试集逐工况配对差值",
        "",
        "下表对每个已知工况分别计算候选方法相对 U-Net 的误差差值；负值表示候选方法误差更低。",
        "",
        *_edge_per_case_table(edge_reports["test"]),
        "",
        *(
            [f"![确定性边界修正及扩散相对 U-Net 的误差变化]({edge_figure})"]
            if edge_figure else
            [f"![边界实验 RMSE 差值]({edge_fallback_figures[0]})",
             f"![边界实验边界误差差值]({edge_fallback_figures[1]})"]
            if len(edge_fallback_figures) >= 2 else []
        ),
        "",
        "扩散单次采样和多次预测均值是两个不同成本点，不能把多次均值的精度与单次推理的成本混为一谈。高温带指标均使用缺失区域内 `T ≥ 1200 K` 的统一定义。",
        "",
        "## 遮挡尺寸与未见工况",
        "",
        "遮挡研究对每个工况分别训练了多尺寸共享 U-Net，并在其上训练残差扩散；它没有测试确定性边界修正。表内扩散 RMSE 是多次采样预测均值，推理时间按 U-Net 一次前向加扩散采样次数估算。",
        "",
        *_mask_rows_table(mask_reports),
        "",
        *(
            [f"![不同遮挡尺寸和测试工况下的缺失区域 RMSE]({mask_figure})"]
            if mask_figure else
            [f"![遮挡尺寸研究的缺失区域 RMSE]({mask_fallback_figure})"]
            if mask_fallback_figure else []
        ),
        "",
        "两个整工况留出分别是 T1050（内部工况）和 T1080（高端工况）；各自测试工况未参与对应模型训练、验证或归一化范围估计。它们是预先选定的两个示例工况，不代表所有未见工况。",
        "",
    ]
    if matched is not None:
        matched_rows, match_info = matched
        lines.extend([
            "## 同帧对照（within-case 测试子集）",
            "",
            f"边界实验测试记录与 within-case 遮挡研究在 `{match_info['frames']}` 个完全相同的帧和一致的温度归一化范围上匹配；边界实验逐帧 JSON 可重算该子集。此表只作同数据条件下的描述性对照：遮挡研究的共享 U-Net 重新按多尺寸训练，架构训练流程与边界实验的冻结 U-Net 不同，因此不能把差异归因于单一算法改动。高温指标使用相同的 1200 K 阈值与非空区域汇总，并同时报告参考热区覆盖、检出率和有效帧数。",
            "",
            *matched_rows,
            "",
        ])
    else:
        lines.extend([
            "## 同帧对照",
            "",
            "边界实验与 within-case 遮挡研究没有可完全匹配的测试帧，故不做跨实验逐帧对照。",
            "",
        ])
    lines.extend([
        "## 推理成本",
        "",
        "边界实验时间来自 JSON 的端到端每帧计时，包含 U-Net 前向以及候选模型所需的修正/扩散过程；测试脚本在 CUDA 上同步计时并做了一次预热。遮挡研究残差扩散时间为 `U-Net 秒/帧 + draws × 残差扩散秒/次`，其脚本未做预热且没有记录预测均值聚合开销。两项研究的硬件、预热和评估面板需核实一致后，才能作精确跨研究速度比较。",
        "",
        *([f"![各实验内部的实测推理耗时]({cost_figure})"] if cost_figure else []),
        "",
        "JSON 记录了所选 checkpoint epoch 与推理时间，但没有完整训练墙钟时间或云 GPU 账单；本报告不估算训练 GPU 小时或租赁费用。扩散需要多个反向采样步骤，实际推理成本也随步数和 draws 增长。",
        "",
        "## 解读范围与限制",
        "",
        "- 训练和评估使用单个随机种子；逐帧观测虽配对，但相邻帧相关，因此不能把帧数当作独立重复，也不报告显著性结论。",
        "- within-case 测试覆盖现有 9 个已知工况的留出帧；未见工况测试只有预选的 T1050 与 T1080 两个整工况。",
        "- 边界实验的 U-Net、确定性边界残差、扩散残差来自一组模型；遮挡研究的 U-Net 和扩散残差来自另一组多遮挡尺寸训练。两条实验线的结果不构成完全控制的直接竞赛。",
        "- 热区指标统一按 `1200 K` 阈值计算，但只统计矩形缺失区域；Dice 对真值与预测均无热区的帧不计入均值，质心偏差要求两侧热区均非空。表中同时给出参考热区覆盖、检测覆盖和 Dice/质心有效帧，避免仅看均值掩盖覆盖不足。",
        "- 测试遮挡尺寸不同会改变任务难度；T1050/T1080 与 within-case 的温度归一化范围来自各自训练池。因此不同折之间的绝对误差不能不加限定地相减。",
        "",
        "## 图表文件",
        "",
    ])
    if figures:
        for figure in figures:
            lines.append(f"- [{figure}]({figure})")
    else:
        lines.append("- 当前 Python 环境没有 Matplotlib，未生成图表；报告表格仍可独立阅读。")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-dir", type=Path, required=True,
                        help="包含 validation.json 和 test.json 的边界实验目录")
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="包含 within-case/T1050/T1080 子目录的遮挡研究目录")
    parser.add_argument("--output", type=Path, required=True,
                        help="输出 Markdown 报告路径；图表保存到同一目录")
    args = parser.parse_args()
    try:
        edge_reports = _load_edge_reports(args.edge_dir)
        mask_reports = _load_mask_reports(args.mask_dir)
        figures = build_report(edge_reports, mask_reports, args.output)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"报告：{args.output}")
    for figure in figures:
        print(f"图表：{args.output.parent / figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
