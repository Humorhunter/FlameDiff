"""Adapt the original training schedule, AE decoder and observation to Azula."""
import math

from azula.denoise import Denoiser, DiracPosterior
from azula.guidance.dps import DPSSampler
from azula.noise import Schedule
import torch


class PartitionMeanDownsample:
    """Full-domain pixel averages on the original integer partition."""

    def __init__(self, shape=(192, 256), low_shape=(16, 21)):
        self.shape = tuple(shape)
        self.low_shape = tuple(low_shape)
        if (len(self.shape) != 2 or len(self.low_shape) != 2
                or any(not isinstance(n, int) or n < 1 for n in (*self.shape, *self.low_shape))
                or any(low > high for high, low in zip(self.shape, self.low_shape))):
            raise ValueError("Expected positive integer HR/LR shapes with LR <= HR.")
        height, width = self.shape
        rows, cols = self.low_shape
        self.row_edges = [i * height // rows for i in range(rows + 1)]
        self.col_edges = [i * width // cols for i in range(cols + 1)]
        self.count = rows * cols
        self._matrices = {}

    def forward(self, field):
        if (field.ndim != 4 or field.shape[1] != 1
                or tuple(field.shape[-2:]) != self.shape or not field.is_floating_point()):
            raise ValueError("Expected a floating Bx1xHxW scalar field.")
        key = (field.device, field.dtype)
        if key not in self._matrices:
            def matrix(edges, size):
                weights = field.new_zeros(len(edges) - 1, size)
                for i, (start, stop) in enumerate(zip(edges, edges[1:])):
                    weights[i, start:stop] = 1 / (stop - start)
                return weights
            self._matrices[key] = (matrix(self.row_edges, self.shape[0]),
                                   matrix(self.col_edges, self.shape[1]))
        rows, cols = self._matrices[key]
        return (rows @ field[:, 0] @ cols.T).flatten(1)


class OriginalSchedule(Schedule):
    def __init__(self, scheduler):
        self.table = torch.cat((torch.ones(1), scheduler.alphas_cumprod.detach().cpu())).double()
        self.n = len(self.table) - 1

    def __call__(self, t):
        t = torch.as_tensor(t)
        if not t.is_floating_point():
            t = t.float()
        if not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
            raise ValueError("Time must lie in [0, 1].")
        # t=(i+1)/1000 maps to DDPM training index i. The clean endpoint is t=0.
        u = t * self.n
        lo = u.floor().long().clamp(0, self.n)
        hi = (lo + 1).clamp(max=self.n)
        table = self.table.to(device=t.device, dtype=t.dtype)
        cumulative = table[lo] + (table[hi] - table[lo]) * (u - lo)
        return cumulative.sqrt(), (1 - cumulative).clamp_min(0).sqrt()


class OriginalEpsilonDenoiser(Denoiser):
    def __init__(self, backbone, scheduler):
        super().__init__()
        self.backbone = backbone
        self.schedule = OriginalSchedule(scheduler)
        self.clip_sample = scheduler.config.clip_sample
        self.clip_range = scheduler.config.clip_sample_range
        if scheduler.config.prediction_type != "epsilon" or scheduler.config.thresholding:
            raise ValueError("Expected original epsilon prediction with no dynamic thresholding.")

    def forward(self, x, t):
        alpha, sigma = self.schedule(t)
        index = (torch.as_tensor(t, device=x.device, dtype=x.dtype) * self.schedule.n - 1)
        index = index.clamp(0, self.schedule.n - 1)
        if index.ndim == 0:
            index = index.expand(len(x))
        epsilon = self.backbone(x, index.reshape(len(x), 1))
        dims = (-1,) + (1,) * (x.ndim - 1)
        clean = (x - sigma.reshape(dims) * epsilon) / alpha.reshape(dims)
        if self.clip_sample:
            clean = clean.clamp(-self.clip_range, self.clip_range)
        return DiracPosterior(mean=clean)


def sample_observations(agent, observations_K, operator, steps=1000, zeta=1.0, seed=0,
                        *, sampler_factory=None, condition=None):
    """Return Bx1x192x256 in K. Optional hooks are supplied by the caller."""
    if steps < 2 or not math.isfinite(zeta) or zeta <= 0:
        raise ValueError("At least two steps and finite positive guidance required.")
    if agent.min_v is None or agent.max_v is None or agent.max_v <= agent.min_v:
        raise ValueError("Explicit original normalization required.")
    device = next(agent.parameters()).device
    values = torch.as_tensor(observations_K, device=device, dtype=torch.float32)
    if (values.ndim != 2 or not len(values) or values.shape[1] != operator.count
            or not torch.isfinite(values).all()):
        raise ValueError("Expected finite nonempty BxM observations in K.")
    if operator.shape != (192, 256):
        raise ValueError("The original AE requires a 192x256 field.")

    sampler_type = DPSSampler if sampler_factory is None else sampler_factory
    parameters = list(agent.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    modes = [(module, module.training) for module in agent.modules()]
    agent.eval().requires_grad_(False)
    width = agent.max_v - agent.min_v
    result = []
    try:
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            for i, observation in enumerate(values):
                torch.manual_seed(seed + i)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed + i)
                target = ((observation - agent.min_v) / width)[None]
                backbone = agent.vitdecoder if condition is None else condition(target)
                denoiser = OriginalEpsilonDenoiser(backbone, agent.scheduler)
                sampler = sampler_type(
                    denoiser, y=target, A=lambda z: operator.forward(agent.autoendecoder.decoder(z)),
                    zeta=float(zeta), steps=steps, start=1.0, stop=0.0, silent=True, device=device,
                )
                state = torch.randn(1, 1, 48, 64, device=device)
                for t, s in sampler.timesteps.unfold(0, 2, 1):
                    state = sampler.step(state, t, s)
                    if not torch.isfinite(state).all():
                        raise ArithmeticError("Nonfinite DPS state.")
                with torch.no_grad():
                    prediction = agent.autoendecoder.decoder(state) * width + agent.min_v
                if not torch.isfinite(prediction).all():
                    raise ArithmeticError("Nonfinite decoded output.")
                result.append(prediction.detach())
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
    return torch.cat(result)
