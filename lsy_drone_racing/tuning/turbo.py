"""TuRBO-1 on the unit cube, after the BoTorch tutorial "BO with TuRBO-1 and TS/qEI".

Eriksson et al., "Scalable Global Optimization via Local Bayesian Optimization", NeurIPS 2019.
A Gaussian process models the reward, and
each round proposes a batch of candidates by Thompson sampling inside a trust region around the
best candidate so far. The trust region grows after repeated improvements, shrinks after repeated
failures, and the search restarts from a new initial design when it has shrunk below a minimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

if TYPE_CHECKING:
    from collections.abc import Sequence

DTYPE = torch.float64


@dataclass(frozen=True)
class TurboState:
    """Trust region of TuRBO-1.

    Attributes:
        dim: Dimension of the search space.
        batch_size: Candidates per round.
        length: Side length of the trust region in the unit cube, before the lengthscale weights.
        length_min: Side length below which the search restarts.
        length_max: Largest side length.
        failure_counter: Rounds in a row without improvement.
        success_counter: Rounds in a row with improvement.
        success_tolerance: Rounds in a row with improvement after which the length doubles.
        best_value: Best reward since the last restart.
        restart_triggered: Whether the length fell below length_min.
    """

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -math.inf
    restart_triggered: bool = False

    @property
    def failure_tolerance(self) -> int:
        """Rounds in a row without improvement after which the length halves."""
        return math.ceil(max(4.0 / self.batch_size, self.dim / self.batch_size))


@dataclass(frozen=True)
class Evaluation:
    """One evaluated candidate.

    Attributes:
        run: Index of the search, which increases with every restart.
        x: The candidate in the unit cube.
        reward: Its reward.
        state: Trust region after the round that evaluated the candidate, None in the initial
            design.
    """

    run: int
    x: tuple[float, ...]
    reward: float
    state: TurboState | None


def update_state(state: TurboState, rewards: Sequence[float]) -> TurboState:
    """Resize the trust region after a round.

    Args:
        state: Trust region before the round.
        rewards: Rewards of the round's candidates.

    Returns:
        The updated trust region.
    """
    best_new = max(rewards)
    if best_new > state.best_value + 1e-3 * math.fabs(state.best_value):
        success_counter, failure_counter = state.success_counter + 1, 0
    else:
        success_counter, failure_counter = 0, state.failure_counter + 1
    length = state.length
    if success_counter == state.success_tolerance:
        length = min(2.0 * length, state.length_max)
        success_counter = 0
    elif failure_counter == state.failure_tolerance:
        length = length / 2.0
        failure_counter = 0
    return replace(
        state,
        length=length,
        failure_counter=failure_counter,
        success_counter=success_counter,
        best_value=max(state.best_value, best_new),
        restart_triggered=length < state.length_min,
    )


def initial_design(dim: int, n: int, seed: int) -> torch.Tensor:
    """Scrambled Sobol points that spread over the unit cube.

    Args:
        dim: Dimension of the search space.
        n: Number of points.
        seed: Seed of the scrambling; the same seed gives the same points.

    Returns:
        The points with shape (n, dim).
    """
    return SobolEngine(dim, scramble=True, seed=seed).draw(n).to(DTYPE)


def fit_model(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    """Fit a Gaussian process with a Matern-5/2 kernel and one lengthscale per dimension.

    Args:
        X: Candidates in the unit cube with shape (n, dim).
        Y: Rewards with shape (n,).

    Returns:
        The fitted model, on standardized rewards.
    """
    dim = X.shape[-1]
    # The noise level is learned (BoTorch's default likelihood). An episode's result depends only
    # on its seed and the parameters, but tiny parameter changes can flip single episodes between
    # finishing and crashing, which the model should treat as noise.
    covar_module = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X, Y.unsqueeze(-1), covar_module=covar_module, outcome_transform=Standardize(m=1)
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def trust_region_bounds(
    x_center: torch.Tensor, lengthscales: torch.Tensor, length: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Box around the center, longer along dimensions in which the reward changes slowly.

    Args:
        x_center: Center of the box in the unit cube with shape (dim,).
        lengthscales: Lengthscales of the model with shape (dim,).
        length: Side length before weighting; the weights have a geometric mean of 1.

    Returns:
        Lower and upper bounds with shape (dim,), clamped to the unit cube.
    """
    weights = lengthscales / lengthscales.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    lower = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
    upper = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)
    return lower, upper


def propose_batch(
    X: torch.Tensor, Y: torch.Tensor, state: TurboState, batch_size: int
) -> torch.Tensor:
    """Propose the next candidates by Thompson sampling inside the trust region.

    The sampled points perturb only some dimensions of the best candidate, which works better than
    perturbing all of them in higher dimensions.

    Args:
        X: Evaluated candidates in the unit cube with shape (n, dim).
        Y: Their rewards with shape (n,).
        state: Current trust region.
        batch_size: Number of candidates to propose.

    Returns:
        The candidates with shape (batch_size, dim).
    """
    model = fit_model(X, Y)
    dim = X.shape[-1]
    n_samples = min(5000, max(2000, 200 * dim))
    x_center = X[Y.argmax()].clone()
    lengthscales = model.covar_module.base_kernel.lengthscale.detach().reshape(dim)
    lower, upper = trust_region_bounds(x_center, lengthscales, state.length)

    sobol = SobolEngine(dim, scramble=True)
    perturbed = lower + (upper - lower) * sobol.draw(n_samples).to(DTYPE)
    # Each dimension is perturbed with probability 20 / dim, and at least one per sample
    mask = torch.rand(n_samples, dim) <= min(20.0 / dim, 1.0)
    unperturbed_rows = torch.where(mask.sum(dim=1) == 0)[0]
    mask[unperturbed_rows, torch.randint(0, dim, size=(len(unperturbed_rows),))] = True
    X_samples = x_center.expand(n_samples, dim).clone()
    X_samples[mask] = perturbed[mask]

    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():
        return thompson_sampling(X_samples, num_samples=batch_size)


def next_batch(
    history: Sequence[Evaluation], dim: int, n_init: int, batch_size: int, x_default: torch.Tensor
) -> tuple[torch.Tensor, int, TurboState | None]:
    """Choose the next candidates from the evaluations so far, which allows resuming a search.

    Every run starts with an initial design of n_init Sobol points, seeded with the run's index; the
    first run's design starts with x_default. After the design, TuRBO proposes the candidates. A
    restart starts a new run, whose model ignores the earlier runs.

    Args:
        history: All evaluations so far, in order.
        dim: Dimension of the search space.
        n_init: Size of the initial design of every run.
        batch_size: Candidates per round.
        x_default: Point in the unit cube of the controller's defaults with shape (dim,).

    Returns:
        - The candidates with shape (<= batch_size, dim). Fewer at the end of the initial design.
        - The run's index.
        - The trust region to update with the candidates' rewards, None in the initial design.
    """
    run = history[-1].run if history else 0
    last_state = history[-1].state if history else None
    if last_state is not None and last_state.restart_triggered:
        run += 1
        last_state = None
    run_history = [evaluation for evaluation in history if evaluation.run == run]

    if len(run_history) < n_init:
        design = initial_design(dim, n_init, seed=run)
        if run == 0:
            design[0] = x_default
        return design[len(run_history) : len(run_history) + batch_size], run, None

    X = torch.tensor([evaluation.x for evaluation in run_history], dtype=DTYPE)
    Y = torch.tensor([evaluation.reward for evaluation in run_history], dtype=DTYPE)
    if last_state is None:
        state = TurboState(dim=dim, batch_size=batch_size, best_value=float(Y.max()))
    else:
        state = last_state
    return propose_batch(X, Y, state, batch_size), run, state
