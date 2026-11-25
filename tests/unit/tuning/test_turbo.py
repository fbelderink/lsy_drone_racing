import numpy as np
import pytest
import torch

from lsy_drone_racing.tuning.turbo import (
    Evaluation,
    TurboState,
    initial_design,
    next_batch,
    propose_batch,
    trust_region_bounds,
    update_state,
)

DIM = 2


def toy_reward(x: torch.Tensor) -> torch.Tensor:
    """Maximum 0 at (0.3, 0.7)."""
    return -((x - torch.tensor([0.3, 0.7], dtype=x.dtype)) ** 2).sum(dim=-1)


@pytest.mark.unit
def test_failure_tolerance_from_dim_and_batch():
    assert TurboState(dim=10, batch_size=2).failure_tolerance == 5
    assert TurboState(dim=2, batch_size=4).failure_tolerance == 1


@pytest.mark.unit
def test_trust_region_shrinks_after_failure_tolerance_rounds():
    state = TurboState(dim=4, batch_size=2, best_value=0.0)
    state = update_state(state, [-1.0, -2.0])
    assert state.length == 0.8 and state.failure_counter == 1
    state = update_state(state, [-1.0])
    assert state.length == 0.4 and state.failure_counter == 0


@pytest.mark.unit
def test_trust_region_grows_after_success_tolerance_rounds():
    state = TurboState(dim=4, batch_size=2, best_value=0.0)
    for round_index in range(state.success_tolerance):
        state = update_state(state, [float(round_index + 1)])
    assert state.length == 1.6
    assert state.best_value == state.success_tolerance
    # Capped at length_max
    for round_index in range(state.success_tolerance):
        state = update_state(state, [100.0 + round_index])
    assert state.length == 1.6


@pytest.mark.unit
def test_restart_below_minimum_length():
    state = TurboState(dim=2, batch_size=4, best_value=0.0, length=0.01)
    assert not state.restart_triggered
    state = update_state(state, [-1.0])
    assert state.length < state.length_min and state.restart_triggered


@pytest.mark.unit
def test_trust_region_bounds_scaled_by_lengthscales_and_clamped():
    lb, ub = trust_region_bounds(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 4.0]), length=0.4)
    # Weights normalized to geometric mean 1: [0.5, 2.0]
    torch.testing.assert_close(lb, torch.tensor([0.4, 0.1]))
    torch.testing.assert_close(ub, torch.tensor([0.6, 0.9]))
    lb, ub = trust_region_bounds(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]), length=0.4)
    torch.testing.assert_close(lb, torch.tensor([0.0, 0.8]))
    torch.testing.assert_close(ub, torch.tensor([0.2, 1.0]))


@pytest.mark.unit
def test_initial_design_is_reproducible_and_in_cube():
    design = initial_design(DIM, 8, seed=3)
    assert design.shape == (8, DIM)
    assert torch.all((design >= 0.0) & (design <= 1.0))
    torch.testing.assert_close(design, initial_design(DIM, 8, seed=3))


@pytest.mark.unit
def test_propose_batch_stays_in_small_trust_region():
    torch.manual_seed(0)
    X = initial_design(DIM, 12, seed=0)
    Y = toy_reward(X)
    state = TurboState(dim=DIM, batch_size=3, best_value=float(Y.max()), length=1e-3)
    X_next = propose_batch(X, Y, state, batch_size=3)
    assert X_next.shape == (3, DIM)
    assert torch.all((X_next >= 0.0) & (X_next <= 1.0))
    # Lengthscale ratios are bounded, so no trust-region side exceeds about 0.03 at this length
    assert torch.all((X_next - X[Y.argmax()]).abs() < 0.02)


def evaluations(run: int, xs: torch.Tensor, state: TurboState | None = None) -> list[Evaluation]:
    return [
        Evaluation(run=run, x=tuple(x.tolist()), reward=float(toy_reward(x)), state=state)
        for x in xs
    ]


@pytest.mark.unit
def test_next_batch_starts_with_defaults_then_continues_design():
    x_default = torch.tensor([0.25, 0.25], dtype=torch.float64)
    x_next, run, state = next_batch([], DIM, n_init=5, batch_size=2, x_default=x_default)
    assert run == 0 and state is None
    torch.testing.assert_close(x_next[0], x_default)
    # Resuming in the middle of the design gives its next points
    design = initial_design(DIM, 5, seed=0)
    history = evaluations(0, x_next)
    x_next, run, state = next_batch(history, DIM, n_init=5, batch_size=2, x_default=x_default)
    assert run == 0 and state is None
    torch.testing.assert_close(x_next, design[2:4])
    # The last batch of the design is shortened to the design's size
    history += evaluations(0, x_next)
    x_next, _, _ = next_batch(history, DIM, n_init=5, batch_size=2, x_default=x_default)
    torch.testing.assert_close(x_next, design[4:5])


@pytest.mark.unit
def test_next_batch_after_design_starts_trust_region_from_best_reward():
    x_default = torch.tensor([0.25, 0.25], dtype=torch.float64)
    history = evaluations(0, initial_design(DIM, 6, seed=0))
    x_next, run, state = next_batch(history, DIM, n_init=6, batch_size=2, x_default=x_default)
    assert run == 0 and x_next.shape == (2, DIM)
    assert state == TurboState(dim=DIM, batch_size=2, best_value=max(e.reward for e in history))
    # A state saved with the last evaluations is continued
    saved = TurboState(dim=DIM, batch_size=2, best_value=0.0, length=0.2, failure_counter=1)
    history += evaluations(0, x_next, saved)
    _, _, state = next_batch(history, DIM, n_init=6, batch_size=2, x_default=x_default)
    assert state == saved


@pytest.mark.unit
def test_next_batch_restarts_with_new_design():
    x_default = torch.tensor([0.25, 0.25], dtype=torch.float64)
    history = evaluations(0, initial_design(DIM, 4, seed=0))
    collapsed = TurboState(dim=DIM, batch_size=2, length=0.001, restart_triggered=True)
    history += evaluations(0, initial_design(DIM, 2, seed=7), collapsed)
    x_next, run, state = next_batch(history, DIM, n_init=4, batch_size=2, x_default=x_default)
    assert run == 1 and state is None
    torch.testing.assert_close(x_next, initial_design(DIM, 4, seed=1)[:2])
    assert not np.allclose(x_next[0].numpy(), x_default.numpy())
