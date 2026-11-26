"""Local unit tests for unified three-task spatial latent-MMPS modules."""

from __future__ import annotations

import argparse
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

from model.EnDeCoder import AutoDecoder, AutoEnDecoder
from model.ViTDecoder import ViTDecoder
from spatial import load_autoencoder
from spatial_dps import diffusion_scheduler
from spatial_mmps import mmps_guidance
from spatial_mmps_observation import DEFAULT_SENSOR_COORDS_XY, SpatialObservation
from spatial_mmps_reconstruct import (
    REQUIRED_METRIC_KEYS,
    TASK_DEFAULTS,
    TemperatureReconstructor,
    _file_sha256,
    _load_resume_rows,
    _row_key,
    _write_json_atomic,
    aggregate_metric_rows,
    build_evaluation_report,
    build_scheduler,
    evaluate,
    hot_zone_metrics,
    parse_device,
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

    def test_mask_cache_is_isolated_from_caller_mutation(self) -> None:
        for task in ("sensors", "inpaint", "sr"):
            op = SpatialObservation(task)
            ref = torch.zeros((2, 1, 192, 256))
            m1 = op.mask(ref)
            original = m1.clone()
            m1.fill_(not bool(m1[0, 0, 0, 0].item()))
            m2 = op.mask(ref)
            self.assertTrue(torch.equal(m2, original))


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
        for rec in step_records:
            self.assertIn("clean_out_of_unit_box_fraction", rec)
            self.assertIn("posterior_clipped_fraction", rec)
            self.assertIn("score_to_noise_norm_ratio", rec)
            self.assertGreaterEqual(rec["clean_out_of_unit_box_fraction"], 0.0)
            self.assertLessEqual(rec["clean_out_of_unit_box_fraction"], 1.0)
            self.assertGreaterEqual(rec["posterior_clipped_fraction"], 0.0)
            self.assertLessEqual(rec["posterior_clipped_fraction"], 1.0)
            self.assertGreaterEqual(rec["score_to_noise_norm_ratio"], 0.0)

    def test_forward_ad_jvp_matches_reference_and_handles_constant_measure(self) -> None:
        decoder = AutoDecoder().eval().requires_grad_(False)
        op = SpatialObservation("sr")
        current = torch.randn((1, 1, 48, 64), requires_grad=True)
        clean = 0.8 * current
        obs = torch.zeros((1, 1, 48, 64))

        score, diag = mmps_guidance(
            current,
            clean,
            obs,
            lambda z: op.measure(decoder(z)),
            alpha_bar=0.5,
            sigma=0.1,
            maxiter=2,
        )
        self.assertEqual(tuple(score.shape), (1, 1, 48, 64))
        self.assertTrue(torch.isfinite(score).all().item())
        self.assertGreaterEqual(diag["iterations"], 1.0)

        # Dead-ReLU / zero-derivative measure must produce finite zero score cleanly
        const_obs = torch.ones((1, 4))
        const_score, const_diag = mmps_guidance(
            current,
            clean,
            const_obs,
            lambda z: F.relu(z.flatten(1)[:, :4] - 1e6),
            alpha_bar=0.5,
            sigma=0.1,
            maxiter=2,
        )
        self.assertEqual(float(const_score.abs().max().item()), 0.0)
        self.assertTrue(math.isfinite(const_diag["relative_residual"]))

    def test_vit_decoder_supports_float64(self) -> None:
        model = ViTDecoder(d_model=32, num_heads=4, num_layers=1, num_feats=0).double().eval()
        latent = torch.randn((1, 1, 48, 64), dtype=torch.float64)
        timestep = torch.tensor([[10]], dtype=torch.long)
        with torch.no_grad():
            out = model(latent, None, timestep)
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(tuple(out.shape), (1, 1, 48, 64))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_woodbury_latent_space_matches_observation_space_nonsymmetric(self) -> None:
        gen = torch.Generator().manual_seed(11)
        d, m = 16, 64
        a_mat = torch.randn((m, d), generator=gen, dtype=torch.float64)
        j_mu = torch.randn((d, d), generator=gen, dtype=torch.float64) * 0.25
        current = torch.randn((1, 1, 4, 4), generator=gen, dtype=torch.float64, requires_grad=True)
        clean = (current.reshape(1, d) @ j_mu.T).reshape(1, 1, 4, 4)
        obs = torch.randn((1, m), generator=gen, dtype=torch.float64)
        measure = lambda z: z.reshape(1, d) @ a_mat.T

        score_obs, diag_obs = mmps_guidance(
            current, clean, obs, measure, alpha_bar=0.5, sigma=0.3, maxiter=64,
            solver_space="observation", verify_residual=True,
        )
        score_lat, diag_lat = mmps_guidance(
            current, clean, obs, measure, alpha_bar=0.5, sigma=0.3, maxiter=64,
            solver_space="latent", verify_residual=False,
        )
        score_auto, diag_auto = mmps_guidance(
            current, clean, obs, measure, alpha_bar=0.5, sigma=0.3, maxiter=64,
            solver_space="auto",
        )
        self.assertEqual(diag_obs["solver_space"], "observation")
        self.assertEqual(diag_lat["solver_space"], "latent")
        self.assertEqual(diag_auto["solver_space"], "latent")
        self.assertLess(float((score_obs - score_lat).abs().max().item()), 1e-11)
        self.assertTrue(torch.allclose(score_lat, score_auto, atol=1e-12))
        self.assertEqual(diag_lat["converged"], 1.0)
        self.assertLess(diag_lat["relative_residual"], 1e-5)

        # When m <= d, auto picks observation space
        small_obs = torch.randn((1, 8), generator=gen, dtype=torch.float64)
        _, diag_small = mmps_guidance(
            current, clean, small_obs, lambda z: z.reshape(1, d)[:, :8],
            alpha_bar=0.5, sigma=0.3, maxiter=8, solver_space="auto",
        )
        self.assertEqual(diag_small["solver_space"], "observation")

    def test_guidance_max_timestep_score_clip_and_ddim(self) -> None:
        op = SpatialObservation("sensors")
        obs = torch.linspace(0.1, 0.9, 8).unsqueeze(0)
        diag: list[dict] = []

        out = sample_mmps(
            self.denoiser,
            self.decoder,
            op.measure,
            obs,
            scheduler=build_scheduler(20, sampler="ddim", clip_sample=True),
            steps=4,
            sigma=0.05,
            maxiter=4,
            seed=9,
            guidance_max_timestep=10,
            score_clip=0.05,
            eta=0.0,
            diagnostics=diag.append,
        )
        self.assertEqual(tuple(out.shape), (1, 1, 192, 256))
        step_records = [r for r in diag if r["phase"] == "step"]
        self.assertEqual(len(step_records), 4)
        self.assertTrue(any(not r["guided"] for r in step_records))
        self.assertTrue(any(r["guided"] for r in step_records))
        for rec in step_records:
            if not rec["guided"]:
                self.assertEqual(rec["iterations"], 0.0)
                self.assertEqual(rec["score_to_noise_norm_ratio"], 0.0)
            else:
                self.assertLessEqual(rec["score_to_noise_norm_ratio"], 0.05 + 1e-6)

        # Deterministic DDIM (eta=0.0) reproduces identically even with fresh seed
        out_again = sample_mmps(
            self.denoiser,
            self.decoder,
            op.measure,
            obs,
            scheduler=build_scheduler(20, sampler="ddim", clip_sample=True),
            steps=4,
            sigma=0.05,
            maxiter=4,
            seed=9,
            guidance_max_timestep=10,
            score_clip=0.05,
            eta=0.0,
        )
        self.assertTrue(torch.equal(out, out_again))

        # Warm-start DDIM must also work without custom timesteps kwarg error
        sr_op = SpatialObservation("sr")
        sr_out = sample_mmps(
            self.denoiser,
            self.decoder,
            sr_op.measure,
            torch.full((1, 1, 48, 64), 0.3),
            scheduler=build_scheduler(20, sampler="ddim", clip_sample=True),
            steps=4,
            sigma=0.05,
            maxiter=2,
            seed=9,
            initial_latent=torch.zeros((1, 1, 48, 64)),
            strength=0.5,
            eta=0.0,
        )
        self.assertEqual(tuple(sr_out.shape), (1, 1, 192, 256))
        self.assertTrue(torch.isfinite(sr_out).all().item())

    def test_multichannel_ae_vit_and_mmps_end_to_end(self) -> None:
        ae4 = AutoEnDecoder(latent_channels=4, output_activation="linear").eval().requires_grad_(False)
        field = torch.randn((1, 1, 192, 256))
        with torch.no_grad():
            z4 = ae4.encoder(field)
            rec = ae4.decoder(z4)
        self.assertEqual(tuple(z4.shape), (1, 4, 48, 64))
        self.assertEqual(tuple(rec.shape), (1, 1, 192, 256))

        # Round-trip load_autoencoder with explicit ae_config and inferred latent_channels
        loaded_cfg = load_autoencoder(
            ae4.state_dict(),
            torch.device("cpu"),
            ae_config={"latent_channels": 4, "output_activation": "linear"},
        )
        self.assertEqual(loaded_cfg.latent_channels, 4)
        self.assertEqual(loaded_cfg.output_activation, "linear")
        loaded_inferred = load_autoencoder(ae4.state_dict(), torch.device("cpu"))
        self.assertEqual(loaded_inferred.latent_channels, 4)
        self.assertEqual(loaded_inferred.output_activation, "relu")

        # Legacy 1-channel default remains intact
        ae1 = AutoEnDecoder().eval().requires_grad_(False)
        loaded1 = load_autoencoder(ae1.state_dict(), torch.device("cpu"))
        self.assertEqual(loaded1.latent_channels, 1)
        self.assertEqual(loaded1.output_activation, "relu")

        prior4 = ViTDecoder(
            d_model=32, num_heads=4, num_layers=1, num_feats=0, in_channels=4
        ).eval().requires_grad_(False)

        # Mismatched channel count between prior and decoder must fail fast
        with self.assertRaises(ValueError):
            sample_mmps(
                prior4, ae1.decoder, SpatialObservation("sensors").measure,
                torch.zeros((1, 8)), scheduler=diffusion_scheduler(20), steps=2, maxiter=2,
            )

        # 4-channel sensors (pure noise) -> auto picks observation space (8 <= 12288)
        diag_s: list[dict] = []
        out_s = sample_mmps(
            prior4, loaded_cfg.decoder, SpatialObservation("sensors").measure,
            torch.zeros((1, 8)), scheduler=diffusion_scheduler(20), steps=2, maxiter=2,
            seed=1, diagnostics=diag_s.append,
        )
        self.assertEqual(tuple(out_s.shape), (1, 1, 192, 256))
        self.assertEqual(
            [r["solver_space"] for r in diag_s if r["phase"] == "step"],
            ["observation", "observation"],
        )

        # 4-channel SR (warm start) -> auto picks observation space (3072 <= 12288)
        diag_sr: list[dict] = []
        out_sr = sample_mmps(
            prior4, loaded_cfg.decoder, SpatialObservation("sr").measure,
            torch.zeros((1, 1, 48, 64)), scheduler=diffusion_scheduler(20), steps=2, maxiter=2,
            seed=2, initial_latent=z4, strength=0.5, diagnostics=diag_sr.append,
        )
        self.assertEqual(tuple(out_sr.shape), (1, 1, 192, 256))
        self.assertEqual(
            [r["solver_space"] for r in diag_sr if r["phase"] == "step"],
            ["observation"],
        )

        # 4-channel inpaint (warm start) -> auto picks latent space (36864 > 12288)
        inp_op = SpatialObservation("inpaint")
        indices = inp_op.mask().flatten().nonzero().flatten()
        diag_inp: list[dict] = []
        out_inp = sample_mmps(
            prior4, loaded_cfg.decoder, lambda f: f.flatten(1).index_select(1, indices),
            torch.zeros((1, indices.numel())), scheduler=diffusion_scheduler(20), steps=2, maxiter=2,
            seed=3, initial_latent=z4, strength=0.5, diagnostics=diag_inp.append,
        )
        self.assertEqual(tuple(out_inp.shape), (1, 1, 192, 256))
        self.assertEqual(
            [r["solver_space"] for r in diag_inp if r["phase"] == "step"],
            ["latent"],
        )

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
                "nmae_pct",
                "rmse_k",
                "gradient_mae_k_per_grid_cell",
                "boundary_gradient_mae_k_per_grid_cell",
                "ssim",
                "observation_rmse_k",
                "hf_energy_ratio_pred",
                "hf_energy_ratio_gt",
                "log_spectrum_rel_error_pct",
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

    def test_parse_device(self) -> None:
        for valid in ("cpu", "mps", "cuda", "cuda:0", "cuda:12"):
            self.assertEqual(parse_device(valid), valid)
        for invalid in ("tpu", "cuda:-1", "cuda:abc", "gpu", "cuda:"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_device(invalid)

    def test_task_defaults_and_disable_guidance_cap(self) -> None:
        self.assertEqual(TASK_DEFAULTS["sensors"]["guidance_max_timestep"], 500)
        self.assertEqual(TASK_DEFAULTS["sensors"]["sigma_k"], 100.0)
        self.assertEqual(TASK_DEFAULTS["inpaint"]["warm_strength"], 0.0)
        self.assertEqual(TASK_DEFAULTS["inpaint"]["maxiter"], 16)
        self.assertEqual(TASK_DEFAULTS["inpaint"]["guidance_max_timestep"], 700)
        self.assertEqual(TASK_DEFAULTS["sr"]["steps"], 50)
        self.assertEqual(TASK_DEFAULTS["sr"]["maxiter"], 16)

        tiny_prior = ViTDecoder(d_model=16, num_heads=4, num_layers=1, num_feats=0)
        fake_ckpt = {
            "model_config": {"d_model": 16, "num_heads": 4, "num_layers": 1, "num_feats": 0},
            "temperature_range": [300.0, 1600.0],
            "weights": tiny_prior.state_dict(),
            "ae_weights": {},
            "diffusion_steps": 1000,
        }
        with patch("spatial_mmps_reconstruct.load_autoencoder", return_value=nn.Identity()):
            # Default sensors picks validated defaults
            args_default = SimpleNamespace(
                task="sensors", device="cpu", seed=0, steps=None, maxiter=None,
                sigma_k=None, warm_strength=None, guidance_max_timestep=None,
            )
            rec = TemperatureReconstructor(args_default, fake_ckpt)
            self.assertEqual(rec.steps, 20)
            self.assertEqual(rec.maxiter, 8)
            self.assertEqual(rec.sigma_k, 100.0)
            self.assertEqual(rec.strength, 0.0)
            self.assertEqual(rec.guidance_max_timestep, 500)

            # Passing guidance_max_timestep=-1 explicitly disables the default cap
            args_uncapped = SimpleNamespace(
                task="sensors", device="cpu", seed=0, steps=None, maxiter=None,
                sigma_k=None, warm_strength=None, guidance_max_timestep=-1,
            )
            rec_uncapped = TemperatureReconstructor(args_uncapped, fake_ckpt)
            self.assertIsNone(rec_uncapped.guidance_max_timestep)

    def test_aggregate_and_resume_compatibility(self) -> None:
        # Union of keys across rows must preserve keys appearing only in later rows
        agg = aggregate_metric_rows([
            {"rmse_k": 10.0, "hot_iou_1200k": None},
            {"rmse_k": 20.0, "hot_iou_1200k": 0.8, "extra_metric": 3.0},
        ])
        self.assertAlmostEqual(agg["rmse_k"], 15.0)
        self.assertAlmostEqual(agg["hot_iou_1200k"], 0.8)
        self.assertAlmostEqual(agg["extra_metric"], 3.0)

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
                "sampler": "ddpm",
                "ddim_eta": None,
                "clip_sample": True,
                "solver_space": "auto",
                "guidance_max_timestep": None,
                "score_clip": None,
            }
            full_metrics = {key: (None if "1200k" in key or "boundary" in key else 1.0) for key in REQUIRED_METRIC_KEYS}
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
                "metrics": dict(full_metrics),
                "raw_metrics": dict(full_metrics),
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
                target_keys={_row_key(sample_row, 0), ("T1040", "s0", 150, 1)},
            )
            self.assertEqual(len(resumed), 1)
            self.assertIn(_row_key(sample_row, 0), resumed)

            # Refuses to truncate when existing file has completed frames outside target_keys
            with self.assertRaises(ValueError):
                _load_resume_rows(
                    out_path,
                    expected_settings=settings,
                    checkpoint_sha256=ckpt_sha,
                    epoch=19,
                    target_keys={("T1040", "s0", 150, 1)},
                )

            # Mismatched settings / sampler / SHA must raise
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
                    expected_settings={**settings, "sampler": "ddim"},
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

            # Legacy file lacking checkpoint_sha256 or incomplete metric schema must raise
            legacy_path = tmp / "legacy.json"
            legacy_report = dict(report)
            legacy_report.pop("checkpoint_sha256")
            _write_json_atomic(legacy_path, legacy_report)
            with self.assertRaises(ValueError):
                _load_resume_rows(
                    legacy_path,
                    expected_settings=settings,
                    checkpoint_sha256=ckpt_sha,
                    epoch=19,
                )

            incomplete_path = tmp / "incomplete.json"
            incomplete_row = dict(sample_row, metrics={"rmse_k": 100.0, "mae_k": 80.0})
            incomplete_report = dict(report, per_frame=[incomplete_row])
            _write_json_atomic(incomplete_path, incomplete_report)
            with self.assertRaises(ValueError):
                _load_resume_rows(
                    incomplete_path,
                    expected_settings=settings,
                    checkpoint_sha256=ckpt_sha,
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


class TestMultiFieldAndBaseFixes(unittest.TestCase):
    """Tests for 3-field (T, OH, HO2) support, channel-selective MMPS, latent standardization, and spectral/CRPS metrics."""

    def test_multifield_observation_adjoint_and_project_all_modes(self) -> None:
        torch.manual_seed(11)
        field = torch.randn((2, 3, 192, 256), dtype=torch.float64)
        for task in ("sensors", "inpaint", "sr"):
            for obs_ch in ((0, 1, 2), (0,), (1,), (0, 2)):
                op = SpatialObservation(task, num_channels=3, observed_channels=obs_ch)
                y = op.measure(field)
                self.assertEqual(tuple(y.shape), (2, *op.shape))
                # Exact Euclidean adjoint <Ax, v> == <x, A^T v>
                v = torch.randn_like(y)
                lhs = float((y * v).sum().item())
                rhs = float((field * op.adjoint(v)).sum().item())
                self.assertAlmostEqual(lhs, rhs, places=9, msg=f"{task} obs_ch={obs_ch}")

                # Projection satisfies A(P(x, y)) == y and leaves unobserved channels untouched
                other = torch.randn_like(field)
                projected = op.project(other, y)
                self.assertTrue(torch.allclose(op.measure(projected), y, atol=1e-10))
                for ch in range(3):
                    if ch not in obs_ch:
                        self.assertTrue(torch.equal(projected[:, ch], other[:, ch]))

    def test_linear_multifield_ae_zero_dead_relu_and_latent_scale(self) -> None:
        from model.EnDeCoder import AutoEnDecoder
        from spatial import compute_latent_scale

        torch.manual_seed(7)
        ae = AutoEnDecoder(
            in_channels=3,
            latent_channels=2,
            output_activation="linear",
            latent_activation="linear",
        ).eval()
        # Cold-zero input (normalized T=0, OH=0, HO2=0) must retain non-zero gradients with linear output
        x = torch.zeros((1, 3, 192, 256), dtype=torch.float32)
        z = ae.encoder(x).detach().requires_grad_(True)
        out = ae.decoder(z)
        self.assertEqual(tuple(out.shape), (1, 3, 192, 256))
        grad_z, = torch.autograd.grad(out[:, 0, 95, 20].sum(), z)
        self.assertGreater(float(grad_z.norm().item()), 1e-6)

        # Latent standardization produces exact zero-mean unit-variance per channel
        batch = torch.randn((4, 3, 192, 256)) * 0.2 + 0.3
        scale = compute_latent_scale(ae, [batch], torch.device("cpu"))
        self.assertEqual(len(scale["mean"]), 2)
        self.assertEqual(len(scale["std"]), 2)
        with torch.no_grad():
            z_raw = ae.encoder(batch)
            mean_t = torch.tensor(scale["mean"]).view(1, 2, 1, 1)
            std_t = torch.tensor(scale["std"]).view(1, 2, 1, 1)
            z_std = (z_raw - mean_t) / std_t
            self.assertAlmostEqual(float(z_std.mean().item()), 0.0, places=4)
            self.assertAlmostEqual(float(z_std.std(unbiased=False).item()), 1.0, places=3)

    def test_three_field_reconstructor_matched_and_t_only_all_tasks(self) -> None:
        from model.EnDeCoder import AutoEnDecoder
        from model.ViTDecoder import ViTDecoder
        from spatial import CHECKPOINT_VERSION

        torch.manual_seed(3)
        ae = AutoEnDecoder(
            in_channels=3,
            latent_channels=2,
            output_activation="linear",
            latent_activation="linear",
        ).eval()
        prior = ViTDecoder(d_model=16, num_heads=2, num_layers=1, num_feats=0, in_channels=2).eval()
        ckpt = {
            "version": CHECKPOINT_VERSION,
            "stage": "prior",
            "epoch": 1,
            "fields": ["T", "OH", "HO2"],
            "field_ranges": {
                "T": [300.0, 1600.0],
                "OH": [0.0, 2.5e-3],
                "HO2": [0.0, 1.5e-4],
            },
            "diffusion_steps": 20,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": False,
            "latent_scale": {"mean": [0.05, -0.05], "std": [0.25, 0.30]},
            "ae_config": {
                "in_channels": 3,
                "latent_channels": 2,
                "output_activation": "linear",
                "latent_activation": "linear",
            },
            "ae_weights": ae.state_dict(),
            "model_config": {"d_model": 16, "num_heads": 2, "num_layers": 1, "num_feats": 0, "in_channels": 2},
            "weights": prior.state_dict(),
        }
        target = torch.stack([
            torch.full((192, 256), 1100.0),
            torch.full((192, 256), 5e-4),
            torch.full((192, 256), 2e-5),
        ], dim=0).unsqueeze(0)
        target = target + torch.randn_like(target) * torch.tensor([50.0, 1e-4, 5e-6]).view(1, 3, 1, 1)

        for task, obs_fields in (
            ("sensors", "T,OH,HO2"),
            ("sensors", "T"),
            ("inpaint", "T"),
            ("sr", "T,OH,HO2"),
            ("sr", "T"),
        ):
            args = SimpleNamespace(
                task=task,
                observed_fields=obs_fields,
                device="cpu",
                seed=0,
                steps=2,
                maxiter=2,
                sigma_k=None,
                sigma_norm=0.05,
                warm_strength=None,
            )
            rec = TemperatureReconstructor(args, ckpt)
            self.assertTrue(rec.clip_sample)
            self.assertAlmostEqual(rec.clip_sample_range, 4.0)
            self.assertEqual(rec.timestep_spacing, "trailing")
            self.assertEqual(rec.beta_schedule, "squaredcos_cap_v2")
            obs = rec.operator.measure(target)
            proj, raw, diag = rec.reconstruct(obs, seed=0)
            self.assertEqual(tuple(proj.shape), (1, 3, 192, 256))
            self.assertEqual(tuple(raw.shape), (1, 3, 192, 256))
            self.assertAlmostEqual(diag["observation_rmse_norm_after_projection"], 0.0, places=4)
            m = rec.metrics(proj, target, obs)
            self.assertIn("per_channel", m)
            self.assertEqual(set(m["per_channel"].keys()), {"T", "OH", "HO2"})
            for fn in ("T", "OH", "HO2"):
                self.assertIn("nmae_pct", m["per_channel"][fn])
                self.assertIn("log_spectrum_rel_error_pct", m["per_channel"][fn])
            # Unobserved channels have observation_rmse=None
            if obs_fields == "T":
                self.assertIsNone(m["per_channel"]["OH"]["observation_rmse"])
                self.assertIsNone(m["per_channel"]["HO2"]["observation_rmse"])

    def test_spectral_and_crps_metrics_exact(self) -> None:
        from spatial_mmps_reconstruct import gaussian_crps, spectral_metrics

        torch.manual_seed(0)
        gt = torch.randn((192, 256))
        m_exact = spectral_metrics(gt, gt)
        self.assertAlmostEqual(m_exact["log_spectrum_rel_error_pct"], 0.0, places=6)
        self.assertAlmostEqual(m_exact["hf_energy_ratio_pred"], m_exact["hf_energy_ratio_gt"], places=6)

        mu = gt.unsqueeze(0).unsqueeze(0)
        std = torch.full_like(mu, 0.1)
        region = torch.ones_like(mu, dtype=torch.bool)
        crps_good = gaussian_crps(mu, std, mu, region)
        crps_bad = gaussian_crps(mu + 2.0, std, mu, region)
        self.assertLess(crps_good, crps_bad)

    def test_mscae_and_latent_unet_end_to_end_mmps_and_dps(self) -> None:
        from model.EnDeCoder import MultiScaleAutoEnDecoder, build_autoencoder
        from model.LatentUNet import LatentUNet
        from spatial import _latent_tv_loss, load_prior_model
        from spatial_mmps_reconstruct import TemperatureReconstructor

        torch.manual_seed(7)
        ae_cfg = {
            "arch": "mscae",
            "in_channels": 3,
            "latent_channels": 1,
            "output_activation": "linear",
            "latent_activation": "linear",
            "base_channels": 12,
        }
        ae = build_autoencoder(ae_cfg).eval()
        self.assertIsInstance(ae, MultiScaleAutoEnDecoder)
        x = torch.randn((1, 3, 192, 256))
        z = ae.encoder(x)
        self.assertEqual(tuple(z.shape), (1, 1, 48, 64))
        self.assertGreater(float(_latent_tv_loss(z).item()), 0.0)
        self.assertAlmostEqual(float(_latent_tv_loss(torch.ones_like(z)).item()), 0.0, places=6)

        # Verify zero dead-ReLU on MSCAE decoder
        z_probe = torch.randn((1, 1, 48, 64), requires_grad=True)
        out_probe = ae.decoder(z_probe)
        self.assertEqual(tuple(out_probe.shape), (1, 3, 192, 256))
        op_s = SpatialObservation("sensors", num_channels=3)
        meas = op_s.measure(out_probe)
        for c in range(3):
            for s in range(8):
                grad, = torch.autograd.grad(meas[0, c, s], z_probe, retain_graph=True)
                self.assertGreater(float(grad.abs().sum().item()), 0.0)

        # LatentUNet supports both float32 and float64 + matches ViTDecoder interface
        unet_cfg = {
            "in_channels": 1,
            "base_channels": 8,
            "channel_mults": [1, 1, 2, 2],
            "num_res_blocks": 1,
            "attention_levels": [2, 3],
            "num_heads": 2,
        }
        unet = LatentUNet(**unet_cfg).eval()
        eps32 = unet(z, None, torch.tensor([[5]]))
        self.assertEqual(tuple(eps32.shape), tuple(z.shape))
        unet64 = LatentUNet(**unet_cfg).double().eval()
        eps64 = unet64(z.double(), None, torch.tensor([[5]]))
        self.assertEqual(eps64.dtype, torch.float64)

        ckpt = {
            "version": 1,
            "stage": "prior",
            "fields": ["T", "OH", "HO2"],
            "field_ranges": {"T": [300.0, 1600.0], "OH": [0.0, 2.5e-3], "HO2": [0.0, 1.5e-4]},
            "channel_stats": {
                "T": {"norm_std": 0.15, "std": 200.0},
                "OH": {"norm_std": 0.12, "std": 3e-4},
                "HO2": {"norm_std": 0.08, "std": 1.2e-5},
            },
            "prior_arch": "unet",
            "model_config": unet_cfg,
            "weights": unet.state_dict(),
            "ae_config": ae_cfg,
            "ae_weights": ae.state_dict(),
            "latent_scale": {"mean": [0.0], "std": [1.0]},
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": False,
            "diffusion_steps": 10,
            "epoch": 1,
        }
        loaded = load_prior_model(ckpt, torch.device("cpu"))
        self.assertIsInstance(loaded, LatentUNet)

        target = torch.stack(
            [
                torch.full((192, 256), 950.0),
                torch.full((192, 256), 5e-4),
                torch.full((192, 256), 3e-5),
            ],
            dim=0,
        ).unsqueeze(0)

        for mode, task, obs_f in (
            ("mmps", "sensors", "T,OH,HO2"),
            ("dps", "sensors", "T,OH,HO2"),
            ("dps", "inpaint", "T,OH,HO2"),
            ("dps", "sr", "T"),
        ):
            args = SimpleNamespace(
                task=task,
                observed_fields=obs_f,
                device="cpu",
                seed=0,
                steps=2,
                maxiter=2,
                sigma_k=None,
                sigma_norm=0.05,
                warm_strength=None,
                guidance_mode=mode,
                dps_weight=0.30,
                dps_channel_balance=True,
            )
            rec = TemperatureReconstructor(args, ckpt)
            self.assertEqual(rec.prior_arch, "unet")
            self.assertEqual(rec.guidance_mode, mode)
            obs = rec.operator.measure(target)
            proj, raw, diag = rec.reconstruct(obs, seed=0)
            self.assertEqual(tuple(proj.shape), (1, 3, 192, 256))
            self.assertTrue(torch.isfinite(proj).all().item())
            self.assertAlmostEqual(diag["observation_rmse_norm_after_projection"], 0.0, places=4)
            if mode == "dps":
                self.assertEqual(diag["solver_space_used"], "adaptive_dps")

    def test_leave_one_case_out_split_zero_leakage(self) -> None:
        import numpy as np
        from spatial_data import DATA_DIRECTORY, FRAME_COUNT, FRAME_SHAPE, TemperatureData
        from spatial_mmps_reconstruct import evaluation_panel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / DATA_DIRECTORY
            for case, base_val in (("T1030", 400.0), ("T1050", 900.0)):
                sdir = root / case / "grade_192_256_npdata_1"
                sdir.mkdir(parents=True)
                arr_t = np.full((FRAME_COUNT, *FRAME_SHAPE), base_val, dtype=np.float32)
                arr_t[:, 0, 0] = base_val + 100.0
                np.save(sdir / "T.npy", arr_t)

            data = TemperatureData(tmp, fields=("T",), holdout_cases=("T1050",))
            # Training range MUST reflect only T1030 [400, 500], NOT T1050 [900, 1000]!
            lo, hi = data.training_range()
            self.assertAlmostEqual(lo, 400.0)
            self.assertAlmostEqual(hi, 500.0)
            manifest = data.split_manifest()
            self.assertEqual(manifest["holdout_cases"], ["T1050"])
            self.assertEqual([f["case"] for f in manifest["files"]], ["T1030"])

            ds_holdout, samples_holdout = evaluation_panel(data, "holdout", (lo, hi))
            self.assertEqual(len(ds_holdout), 1)
            self.assertEqual(samples_holdout[0]["case"], "T1050")
            # Holdout frame normalized by T1030 range (400..500): (900-400)/100 = 5.0
            self.assertAlmostEqual(float(ds_holdout[0][0, 1, 1].item()), 5.0, places=4)

    def test_channel_whitened_and_adaptive_mmps_exact(self) -> None:
        from model.EnDeCoder import build_autoencoder
        from model.LatentUNet import LatentUNet
        from spatial_mmps import mmps_guidance
        from spatial_mmps_reconstruct import TemperatureReconstructor

        torch.manual_seed(11)
        # 1. Verify mathematical identity: whitened MMPS (W A, W y, sigma) matches explicit
        # anisotropic covariance Sigma_y = sigma^2 W^{-2} in BOTH observation and latent spaces!
        cur = torch.randn((1, 2, 4, 4), dtype=torch.float64, requires_grad=True)
        W_lin = torch.randn((2 * 16, 2 * 16), dtype=torch.float64) * 0.2
        clean = (cur.reshape(1, -1) @ W_lin.T).reshape(1, 2, 4, 4)
        A = torch.randn((6, 2 * 16), dtype=torch.float64)
        w = torch.tensor([0.5, 0.5, 1.0, 1.0, 2.5, 2.5], dtype=torch.float64)  # diagonal whitening
        y = torch.randn((1, 6), dtype=torch.float64)
        sigma = 0.25
        alpha_bar = 0.6

        def meas_w(z):
            return (z.reshape(1, -1) @ A.T) * w.unsqueeze(0)

        score_obs, d_obs = mmps_guidance(
            cur, clean, y * w.unsqueeze(0), meas_w, alpha_bar,
            sigma=sigma, maxiter=6, solver_space="observation",
        )
        score_lat, d_lat = mmps_guidance(
            cur, clean, y * w.unsqueeze(0), meas_w, alpha_bar,
            sigma=sigma, maxiter=32, solver_space="latent",
        )
        self.assertEqual(d_obs["converged"], 1.0)
        self.assertEqual(d_lat["converged"], 1.0)
        self.assertTrue(torch.allclose(score_obs, score_lat, atol=1e-6, rtol=1e-5))

        # Closed-form reference with explicit anisotropic Sigma_y = diag((sigma / w)^2)
        with torch.no_grad():
            cov_scale = (1.0 - alpha_bar) / (alpha_bar ** 0.5)
            S = cov_scale * W_lin.T  # J_mu^T = W_lin^T
            Sigma_y = torch.diag((sigma / w) ** 2)
            r = (y - (clean.reshape(1, -1) @ A.T)).squeeze(0)
            sys_mat = Sigma_y + A @ S @ A.T
            ref = (W_lin.T @ (A.T @ torch.linalg.solve(sys_mat, r))).reshape(1, 2, 4, 4)
        self.assertTrue(torch.allclose(score_obs, ref, atol=1e-6, rtol=1e-5))

        # 2. Verify TemperatureReconstructor full-rank maxiter=24 for 3ch sensors + whitening + adaptive weight
        ae_cfg = {"arch": "mscae", "in_channels": 3, "latent_channels": 2, "output_activation": "linear", "latent_activation": "linear", "base_channels": 12}
        ae = build_autoencoder(ae_cfg).eval()
        unet_cfg = {"in_channels": 2, "base_channels": 8, "channel_mults": [1, 1, 2, 2], "num_res_blocks": 1, "attention_levels": [2, 3], "num_heads": 2}
        unet = LatentUNet(**unet_cfg).eval()
        ckpt = {
            "version": 1, "stage": "prior", "fields": ["T", "OH", "HO2"],
            "field_ranges": {"T": [300.0, 1600.0], "OH": [0.0, 2.5e-3], "HO2": [0.0, 1.5e-4]},
            "channel_stats": {
                "T": {"norm_std": 0.15, "std": 200.0},
                "OH": {"norm_std": 0.12, "std": 3e-4},
                "HO2": {"norm_std": 0.08, "std": 1.2e-5},
            },
            "prior_arch": "unet", "model_config": unet_cfg, "weights": unet.state_dict(),
            "ae_config": ae_cfg, "ae_weights": ae.state_dict(),
            "latent_scale": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "beta_schedule": "squaredcos_cap_v2", "clip_sample": False, "diffusion_steps": 10, "epoch": 1,
        }
        target = torch.stack([torch.full((192, 256), 950.0), torch.full((192, 256), 5e-4), torch.full((192, 256), 3e-5)], dim=0).unsqueeze(0)
        for task in ("sensors", "inpaint", "sr"):
            args = SimpleNamespace(
                task=task, observed_fields="T,OH,HO2", device="cpu", seed=0, steps=2,
                maxiter=None if task == "sensors" else 2, sigma_k=None, sigma_norm=0.05, warm_strength=0.0,
                guidance_mode="mmps", mmps_channel_whiten=True, mmps_adaptive_weight=0.30,
            )
            rec = TemperatureReconstructor(args, ckpt)
            if task == "sensors":
                self.assertEqual(rec.maxiter, 24)
            self.assertTrue(rec.mmps_channel_whiten)
            self.assertAlmostEqual(rec.mmps_adaptive_weight, 0.30)
            obs = rec.operator.measure(target)
            proj, raw, diag = rec.reconstruct(obs, seed=0)
            self.assertTrue(torch.isfinite(proj).all().item())
            self.assertLessEqual(diag["max_score_to_noise_norm_ratio"], 0.30 + 1e-4)


if __name__ == "__main__":
    unittest.main()
