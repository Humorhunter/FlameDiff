"""Local unit tests for unified three-task spatial latent-MMPS modules."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_dps import diffusion_scheduler
from spatial_mmps_observation import DEFAULT_SENSOR_COORDS_XY, SpatialObservation
from spatial_mmps_reconstruct import (
    TemperatureReconstructor,
    _file_sha256,
    _load_resume_rows,
    _row_key,
    _write_json_atomic,
    aggregate_metric_rows,
    build_evaluation_report,
    evaluate,
    hot_zone_metrics,
)
from spatial_mmps_sampler import sample_mmps


class _TinyDenoiser(nn.Module):
    """Lightweight epsilon predictor matching ViTDecoder's call signature."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            self.conv.weight.fill_(0.02)

    def forward(self, latent: torch.Tensor, sensors, timestep: torch.Tensor) -> torch.Tensor:
        scale = 0.05 + 0.0001 * timestep.float().view(-1, 1, 1, 1)
        return self.conv(latent) + scale * latent


class _TinyDecoder(nn.Module):
    """Lightweight differentiable 4x latent-to-field decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.weight.fill_(1.0 / 9.0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(latent, scale_factor=4, mode="bilinear", align_corners=False)
        return self.proj(upsampled)


class TestSpatialObservation(unittest.TestCase):
    def test_adjoint_identity_all_tasks(self) -> None:
        gen = torch.Generator().manual_seed(42)
        for task in ("sensors", "inpaint", "sr"):
            op = SpatialObservation(task)
            x = torch.randn((2, 1, 192, 256), generator=gen, dtype=torch.float64)
            y = torch.randn((2, *op.shape), generator=gen, dtype=torch.float64)
            lhs = torch.sum(op.measure(x) * y)
            rhs = torch.sum(x * op.adjoint(y))
            self.assertTrue(
                torch.allclose(lhs, rhs, rtol=1e-6, atol=1e-6),
                msg=f"Adjoint mismatch for {task}: lhs={lhs.item()}, rhs={rhs.item()}",
            )

    def test_project_consistency_idempotency_and_autograd(self) -> None:
        gen = torch.Generator().manual_seed(7)
        for task in ("sensors", "inpaint", "sr"):
            op = SpatialObservation(task)
            target = torch.randn((2, 1, 192, 256), generator=gen)
            obs = op.measure(target)
            field = torch.randn((2, 1, 192, 256), generator=gen, requires_grad=True)
            projected = op.project(field, obs)
            self.assertEqual(projected.shape, field.shape)

            # A(project(x, y)) == y
            self.assertTrue(torch.allclose(op.measure(projected), obs, atol=1e-5))
            # Idempotent: project(project(x, y), y) == project(x, y)
            self.assertTrue(torch.allclose(op.project(projected, obs), projected, atol=1e-6))
            # Unmeasured pixels untouched for sensors and inpaint
            if task in ("sensors", "inpaint"):
                unmeasured = ~op.mask(field)
                self.assertTrue(torch.equal(projected[unmeasured], field[unmeasured]))
            # Differentiable in field
            projected.square().mean().backward()
            self.assertIsNotNone(field.grad)
            self.assertTrue(torch.isfinite(field.grad).all().item())

    def test_inpaint_nan_inside_hole_allowed_and_ignored(self) -> None:
        op = SpatialObservation("inpaint")
        obs = torch.ones((1, 1, 192, 256))
        obs[..., 48:144, 64:192] = float("nan")
        op.validate_observation(obs, finite=True)

        adj = op.adjoint(obs)
        self.assertTrue(torch.isfinite(adj).all().item())
        self.assertEqual(float(adj[..., 48:144, 64:192].abs().max().item()), 0.0)

        field = torch.full((1, 1, 192, 256), 2.0)
        projected = op.project(field, obs)
        self.assertTrue(torch.isfinite(projected).all().item())
        self.assertTrue(torch.all(projected[..., 48:144, 64:192] == 2.0).item())
        self.assertTrue(torch.all(projected[op.mask(field)] == 1.0).item())

        # NaN outside hole must fail validation
        bad = obs.clone()
        bad[..., 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            op.validate_observation(bad, finite=True)

    def test_sensor_coordinate_and_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            SpatialObservation("unknown")
        # Duplicate sensor coordinate
        dup = list(DEFAULT_SENSOR_COORDS_XY)
        dup[1] = dup[0]
        with self.assertRaises(ValueError):
            SpatialObservation("sensors", sensor_coords_xy=dup)
        # Out-of-bounds sensor coordinate
        oob = list(DEFAULT_SENSOR_COORDS_XY)
        oob[0] = (256, 0)
        with self.assertRaises(ValueError):
            SpatialObservation("sensors", sensor_coords_xy=oob)
        # Wrong count
        with self.assertRaises(ValueError):
            SpatialObservation("sensors", sensor_coords_xy=DEFAULT_SENSOR_COORDS_XY[:7])
        # Bool coordinate rejected
        bool_coords = list(DEFAULT_SENSOR_COORDS_XY)
        bool_coords[0] = (True, 10)
        with self.assertRaises(ValueError):
            SpatialObservation("sensors", sensor_coords_xy=bool_coords)

        op = SpatialObservation("sr")
        with self.assertRaises(TypeError):
            op.measure(torch.zeros((1, 1, 192, 256), dtype=torch.int32))
        with self.assertRaises(ValueError):
            op.measure(torch.zeros((1, 1, 64, 64)))


class TestSampleMMPS(unittest.TestCase):
    def setUp(self) -> None:
        self.denoiser = _TinyDenoiser().eval().requires_grad_(False)
        self.decoder = _TinyDecoder().eval().requires_grad_(False)

    def test_seeded_reproducibility_and_diagnostics_sensors(self) -> None:
        op = SpatialObservation("sensors")
        obs = torch.linspace(0.1, 0.9, 8).unsqueeze(0)
        diag_a: list[dict] = []
        diag_b: list[dict] = []

        out_a = sample_mmps(
            self.denoiser,
            self.decoder,
            op.measure,
            obs,
            scheduler=diffusion_scheduler(20),
            steps=3,
            sigma=0.05,
            maxiter=4,
            seed=123,
            diagnostics=diag_a.append,
        )
        out_b = sample_mmps(
            self.denoiser,
            self.decoder,
            op.measure,
            obs,
            scheduler=diffusion_scheduler(20),
            steps=3,
            sigma=0.05,
            maxiter=4,
            seed=123,
            diagnostics=diag_b.append,
        )
        out_c = sample_mmps(
            self.denoiser,
            self.decoder,
            op.measure,
            obs,
            scheduler=diffusion_scheduler(20),
            steps=3,
            sigma=0.05,
            maxiter=4,
            seed=124,
        )
        self.assertEqual(tuple(out_a.shape), (1, 1, 192, 256))
        self.assertTrue(torch.equal(out_a, out_b))
        self.assertFalse(torch.allclose(out_a, out_c))

        step_records = [r for r in diag_a if r["phase"] == "step"]
        summary_records = [r for r in diag_a if r["phase"] == "summary"]
        self.assertEqual(len(step_records), 3)
        self.assertEqual(len(summary_records), 1)
        self.assertEqual(summary_records[0]["measurement_dimension"], 8)
        self.assertTrue(math.isfinite(summary_records[0]["raw_observation_rmse"]))

    def test_warm_start_inpaint_and_sr(self) -> None:
        init_latent = torch.zeros((1, 1, 48, 64))

        # SR warm start
        sr_op = SpatialObservation("sr")
        sr_obs = torch.full((1, 1, 48, 64), 0.3)
        sr_diag: list[dict] = []
        sr_out = sample_mmps(
            self.denoiser,
            self.decoder,
            sr_op.measure,
            sr_obs,
            scheduler=diffusion_scheduler(20),
            steps=4,
            sigma=0.05,
            maxiter=2,
            seed=5,
            initial_latent=init_latent,
            strength=0.5,
            diagnostics=sr_diag.append,
        )
        self.assertEqual(tuple(sr_out.shape), (1, 1, 192, 256))
        self.assertEqual(len([r for r in sr_diag if r["phase"] == "step"]), 2)

        # Inpaint warm start with compact known-pixel measure
        inp_op = SpatialObservation("inpaint")
        indices = inp_op.mask().flatten().nonzero().flatten()

        def compact_measure(field: torch.Tensor) -> torch.Tensor:
            return field.flatten(1).index_select(1, indices)

        inp_obs = torch.full((1, indices.numel()), 0.2)
        inp_out = sample_mmps(
            self.denoiser,
            self.decoder,
            compact_measure,
            inp_obs,
            scheduler=diffusion_scheduler(20),
            steps=4,
            sigma=0.05,
            maxiter=2,
            seed=6,
            initial_latent=init_latent,
            strength=0.25,
        )
        self.assertEqual(tuple(inp_out.shape), (1, 1, 192, 256))
        self.assertTrue(torch.isfinite(inp_out).all().item())

    def test_validation_and_mha_fastpath_restoration(self) -> None:
        op = SpatialObservation("sensors")
        obs = torch.zeros((1, 8))
        sched = diffusion_scheduler(20)

        # Reject training mode
        self.denoiser.train()
        try:
            with self.assertRaises(ValueError):
                sample_mmps(self.denoiser, self.decoder, op.measure, obs, scheduler=sched, steps=2)
        finally:
            self.denoiser.eval()

        # Reject inference_mode
        with torch.inference_mode():
            with self.assertRaises(RuntimeError):
                sample_mmps(self.denoiser, self.decoder, op.measure, obs, scheduler=sched, steps=2)

        # Reject batch_size != 1
        with self.assertRaises(ValueError):
            sample_mmps(self.denoiser, self.decoder, op.measure, torch.zeros((2, 8)), scheduler=sched, steps=2)

        # Reject partial warm start arguments
        with self.assertRaises(ValueError):
            sample_mmps(
                self.denoiser, self.decoder, op.measure, obs, scheduler=sched, steps=2, strength=0.5
            )

        # Reject both seed and generator
        with self.assertRaises(ValueError):
            sample_mmps(
                self.denoiser,
                self.decoder,
                op.measure,
                obs,
                scheduler=sched,
                steps=2,
                seed=1,
                generator=torch.Generator().manual_seed(1),
            )

        # Restore MHA fastpath state even on error
        orig_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(True)
        try:
            def broken_measure(_field: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("boom")

            with self.assertRaises(RuntimeError):
                sample_mmps(self.denoiser, self.decoder, broken_measure, obs, scheduler=sched, steps=2)
            self.assertTrue(torch.backends.mha.get_fastpath_enabled())
        finally:
            torch.backends.mha.set_fastpath_enabled(orig_fastpath)


class TestMetricsAndResume(unittest.TestCase):
    def _stub_reconstructor(self, task: str) -> TemperatureReconstructor:
        stub = object.__new__(TemperatureReconstructor)
        stub.operator = SpatialObservation(task)
        stub.low = 300.0
        stub.high = 1600.0
        stub.span = 1300.0
        return stub

    def test_hot_zone_metrics_exact_and_empty_threshold(self) -> None:
        target = torch.full((1, 1, 4, 4), 500.0)
        pred = torch.full((1, 1, 4, 4), 500.0)
        region = torch.ones_like(target, dtype=torch.bool)

        # Target hot (>=1000 K): 4 pixels at [0:2, 0:2] = 1100 K
        target[..., 0:2, 0:2] = 1100.0
        # Pred hot (>=1000 K): 4 pixels at [0:2, 1:3] = 1050 K -> intersection=2, union=6
        pred[..., 0:2, 1:3] = 1050.0
        # On target_hot: 2 pixels have pred=500 (err=-600), 2 pixels have pred=1050 (err=-50)
        res = hot_zone_metrics(pred, target, region, thresholds_k=(1000.0, 1200.0))

        expected_mae = (600.0 * 2 + 50.0 * 2) / 4.0
        expected_rmse = math.sqrt((600.0**2 * 2 + 50.0**2 * 2) / 4.0)
        self.assertAlmostEqual(res["hot_mae_1000k_k"], expected_mae, places=4)
        self.assertAlmostEqual(res["hot_rmse_1000k_k"], expected_rmse, places=4)
        self.assertAlmostEqual(res["hot_iou_1000k"], 2.0 / 6.0, places=6)
        self.assertAlmostEqual(res["hot_dice_1000k"], 4.0 / 8.0, places=6)

        # Nothing >= 1200 K -> all None (never NaN)
        self.assertIsNone(res["hot_mae_1200k_k"])
        self.assertIsNone(res["hot_rmse_1200k_k"])
        self.assertIsNone(res["hot_iou_1200k"])
        self.assertIsNone(res["hot_dice_1200k"])

    def test_reconstructor_metrics_all_tasks(self) -> None:
        target = torch.full((1, 1, 192, 256), 400.0)
        target[..., 60:130, 80:170] = 1300.0
        pred = target.clone()
        pred[..., 60:130, 80:170] = 1250.0

        for task in ("sensors", "inpaint", "sr"):
            rec = self._stub_reconstructor(task)
            obs = rec.operator.measure(target)
            projected = rec.operator.project(pred, obs)
            m = rec.metrics(projected, target, obs)

            expected_keys = {
                "mae_k",
                "rmse_k",
                "gradient_mae_k_per_grid_cell",
                "boundary_gradient_mae_k_per_grid_cell",
                "ssim",
                "observation_rmse_k",
                "hot_mae_1000k_k",
                "hot_rmse_1000k_k",
                "hot_iou_1000k",
                "hot_dice_1000k",
                "hot_mae_1200k_k",
                "hot_rmse_1200k_k",
                "hot_iou_1200k",
                "hot_dice_1200k",
            }
            self.assertEqual(set(m.keys()), expected_keys)
            self.assertLess(m["observation_rmse_k"], 1e-4)
            if task == "inpaint":
                self.assertIsNotNone(m["boundary_gradient_mae_k_per_grid_cell"])
            else:
                self.assertIsNone(m["boundary_gradient_mae_k_per_grid_cell"])
            self.assertAlmostEqual(m["hot_iou_1000k"], 1.0, places=6)
            self.assertAlmostEqual(m["hot_dice_1200k"], 1.0, places=6)
            # Must serialize cleanly without NaN
            json.dumps(m, allow_nan=False)

    def test_aggregate_and_resume_compatibility(self) -> None:
        agg = aggregate_metric_rows([
            {"rmse_k": 10.0, "hot_iou_1200k": None},
            {"rmse_k": 20.0, "hot_iou_1200k": 0.8},
        ])
        self.assertAlmostEqual(agg["rmse_k"], 15.0)
        self.assertAlmostEqual(agg["hot_iou_1200k"], 0.8)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt_path = tmp / "prior.pt"
            ckpt_path.write_bytes(b"fake-checkpoint-bytes")
            ckpt_sha = _file_sha256(ckpt_path)
            out_path = tmp / "eval.json"

            settings = {
                "task": "sensors",
                "device": "cpu",
                "temperature_range_k": [300.0, 1600.0],
                "measured_count": 8,
                "steps": 20,
                "maxiter": 8,
                "sigma_k": 65.0,
                "warm_strength": 0.0,
                "seed": 0,
            }
            sample_row = {
                "case": "T1030",
                "slice": "s0",
                "path": "T1030/s0/T.npy",
                "frame": 150,
                "seed": 0,
                "seconds": 1.25,
                "observation_rmse_k_before_projection": 12.0,
                "observation_rmse_k_after_projection": 0.0,
                "solver_steps": 20,
                "solver_converged_steps": 5,
                "solver_median_relative_residual": 1e-4,
                "solver_max_relative_residual": 1e-2,
                "metrics": {"rmse_k": 100.0, "hot_iou_1200k": None},
                "raw_metrics": {"rmse_k": 101.0, "hot_iou_1200k": None},
            }
            report = build_evaluation_report(
                settings=settings,
                checkpoint_path=ckpt_path,
                checkpoint_sha256=ckpt_sha,
                epoch=19,
                task="sensors",
                target_frames=2,
                rows=[sample_row],
                status="running",
            )
            _write_json_atomic(out_path, report)
            self.assertFalse(out_path.with_name(out_path.name + ".tmp").exists())

            resumed = _load_resume_rows(
                out_path,
                expected_settings=settings,
                checkpoint_sha256=ckpt_sha,
                epoch=19,
            )
            self.assertEqual(len(resumed), 1)
            self.assertIn(_row_key(sample_row, 0), resumed)

            # Mismatched settings must raise
            bad_settings = {**settings, "steps": 50}
            with self.assertRaises(ValueError):
                _load_resume_rows(
                    out_path,
                    expected_settings=bad_settings,
                    checkpoint_sha256=ckpt_sha,
                    epoch=19,
                )
            with self.assertRaises(ValueError):
                _load_resume_rows(
                    out_path,
                    expected_settings=settings,
                    checkpoint_sha256="deadbeef",
                    epoch=19,
                )

    def test_evaluate_resume_skips_completed_and_writes_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt_path = tmp / "prior.pt"
            ckpt_path.write_bytes(b"checkpoint-content")
            out_path = tmp / "val.json"
            ckpt_sha = _file_sha256(ckpt_path)

            samples = [
                {"case": "T1030", "slice": "s0", "path": "T1030/s0/T.npy", "frame": 150},
                {"case": "T1040", "slice": "s0", "path": "T1040/s0/T.npy", "frame": 150},
            ]
            # Normalized frames [1, 192, 256]
            dataset = [
                torch.full((1, 192, 256), 0.5),
                torch.full((1, 192, 256), 0.7),
            ]
            reconstruct_calls: list[int] = []

            class _FakeReconstructor:
                def __init__(self, _args, _checkpoint) -> None:
                    self.device = torch.device("cpu")
                    self.low = 300.0
                    self.high = 1600.0
                    self.span = 1300.0
                    self.operator = SpatialObservation("sr")

                def settings(self) -> dict:
                    return {
                        "task": "sr",
                        "device": "cpu",
                        "temperature_range_k": [300.0, 1600.0],
                        "measured_count": 3072,
                        "steps": 4,
                        "maxiter": 2,
                        "sigma_k": 10.0,
                        "warm_strength": 0.1,
                        "seed": 0,
                    }

                def reconstruct(self, observation: torch.Tensor, *, seed: int):
                    reconstruct_calls.append(seed)
                    raw = torch.full((1, 1, 192, 256), 1200.0)
                    proj = self.operator.project(raw, observation)
                    diag = {
                        "observation_rmse_k_before_projection": 5.0,
                        "observation_rmse_k_after_projection": 0.0,
                        "solver_steps": 1,
                        "solver_converged_steps": 1,
                        "solver_median_relative_residual": 1e-6,
                        "solver_max_relative_residual": 1e-6,
                    }
                    return proj, raw, diag

                def observation_rmse(self, field, observation):
                    return TemperatureReconstructor.observation_rmse(self, field, observation)

                def metrics(self, prediction, target, observation):
                    return TemperatureReconstructor.metrics(self, prediction, target, observation)

            # Pre-populate first frame as completed in a partial "running" file
            fake_rec = _FakeReconstructor(None, None)
            t0 = dataset[0].unsqueeze(0) * 1300.0 + 300.0
            o0 = fake_rec.operator.measure(t0)
            p0, r0, d0 = fake_rec.reconstruct(o0, seed=0)
            reconstruct_calls.clear()
            first_row = {
                **samples[0],
                "seed": 0,
                "seconds": 0.1,
                **d0,
                "metrics": fake_rec.metrics(p0, t0, o0),
                "raw_metrics": fake_rec.metrics(r0, t0, o0),
            }
            partial = build_evaluation_report(
                settings=fake_rec.settings(),
                checkpoint_path=ckpt_path,
                checkpoint_sha256=ckpt_sha,
                epoch=19,
                task="sr",
                target_frames=2,
                rows=[first_row],
                status="running",
            )
            _write_json_atomic(out_path, partial)

            args = SimpleNamespace(
                data=tmp,
                checkpoint=ckpt_path,
                output=out_path,
                task="sr",
                device="cpu",
                seed=0,
                steps=4,
                sigma_k=10.0,
                maxiter=2,
                warm_strength=0.1,
                limit=None,
                resume=True,
            )

            class _FakeData:
                def __init__(self, _path) -> None:
                    pass

                def training_range(self):
                    return (300.0, 1600.0)

            with (
                patch("spatial_mmps_reconstruct.TemperatureData", _FakeData),
                patch("spatial_mmps_reconstruct.load_checkpoint", return_value={"epoch": 19}),
                patch("spatial_mmps_reconstruct.check_provenance"),
                patch("spatial_mmps_reconstruct.TemperatureReconstructor", _FakeReconstructor),
                patch("spatial_mmps_reconstruct.validation_panel", return_value=(dataset, samples)),
            ):
                evaluate(args)

            # Only frame index 1 (seed=1) should have been computed
            self.assertEqual(reconstruct_calls, [1])
            final = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["target_frames"], 2)
            self.assertEqual(final["frames"], 2)
            self.assertEqual([row["case"] for row in final["per_frame"]], ["T1030", "T1040"])
            self.assertIn("summary_metrics", final)
            self.assertIn("summary_raw_metrics", final)
            self.assertIn("summary_observation", final)
            self.assertIn("hot_iou_1000k", final["summary_metrics"])
            self.assertFalse(out_path.with_name(out_path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
