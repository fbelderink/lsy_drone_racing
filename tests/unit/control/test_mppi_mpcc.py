from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from crazyflow.dynamics import Dynamics, load_params

from lsy_drone_racing.control.attitude_mppi import discretize
from lsy_drone_racing.control.mpcc.cost_pp import CorridorObjective
from lsy_drone_racing.control.mpcc.mppi_mpcc import (
    COLLISION_WEIGHTS,
    MPPIMPCC,
    V_THETA_MAX,
    ContouringObjective,
    collision_cost,
    crash_mask,
    freeze_after_crash,
    mppi_mpcc_update,
)
from lsy_drone_racing.control.mpcc.path import path_from_reference
from lsy_drone_racing.control.planner.pmm import DensePathReference
from lsy_drone_racing.control.utils.collision import (
    gate_contact,
    obstacle_clearance,
    obstacle_position_uncertainty,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lsy_drone_racing.control.mpcc.path import ArcLengthPath

DRONE_RADIUS = 0.086
# Gate at the origin, 1 m up, facing +x
GATES = jnp.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
LIMITS = jnp.array([[-3.0, -3.0, -1e-3], [3.0, 3.0, 2.0]])


def line(start: list[float], end: list[float], n: int = 11) -> jnp.ndarray:
    """Positions evenly spaced from start to end with shape (n, 3)."""
    return jnp.linspace(jnp.array(start), jnp.array(end), n)


@pytest.mark.unit
def test_obstacle_clearance_nearest_pole_and_inflation():
    pos = jnp.array([[0.5, 0.0, 1.0]])
    obstacles = jnp.array([[0.0, 0.0, 1.5], [2.0, 0.0, 1.5]])
    clearance = obstacle_clearance(pos, obstacles, jnp.zeros(2), radius=0.1)
    np.testing.assert_allclose(clearance, [0.4], atol=1e-6)
    # Inflating the far pole changes nothing, inflating the near one reduces the clearance
    clearance = obstacle_clearance(pos, obstacles, jnp.array([0.0, 0.3]), radius=0.1)
    np.testing.assert_allclose(clearance, [0.4], atol=1e-6)
    clearance = obstacle_clearance(pos, obstacles, jnp.array([0.3, 0.0]), radius=0.1)
    np.testing.assert_allclose(clearance, [0.1], atol=1e-6)


@pytest.mark.unit
def test_obstacle_clearance_ignores_height():
    obstacles = jnp.array([[0.0, 0.0, 1.5]])
    low = obstacle_clearance(jnp.array([[0.5, 0.0, 0.2]]), obstacles, jnp.zeros(1), radius=0.1)
    high = obstacle_clearance(jnp.array([[0.5, 0.0, 1.9]]), obstacles, jnp.zeros(1), radius=0.1)
    np.testing.assert_allclose(low, high)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("pos", "expected"),
    [
        ([0.0, 0.0, 1.0], False),  # Center of the opening
        ([0.0, 0.1, 1.0], False),  # Inside the opening, clear of the frame
        ([0.0, 0.25, 1.0], True),  # On the side bar
        ([0.0, 0.0, 1.3], True),  # On the top bar
        ([0.0, 0.5, 1.0], False),  # Beside the frame
        ([0.5, 0.25, 1.0], False),  # In front of the side bar
        ([0.0, 0.0, 0.3], True),  # On the stand
        ([0.0, 0.3, 0.3], False),  # Beside the stand
    ],
)
def test_gate_contact(pos: list[float], expected: bool):
    assert bool(gate_contact(jnp.array([pos]), GATES, DRONE_RADIUS)[0]) == expected


@pytest.mark.unit
def test_crash_mask_obstacle_stays_set():
    pos = line([-1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    obstacles = jnp.array([[0.0, 1.0, 1.5]])
    clearance = obstacle_clearance(pos, obstacles, jnp.zeros(1), radius=0.1)
    crashed = crash_mask(pos, clearance, GATES, LIMITS, DRONE_RADIUS)
    # Steps at x = -1.0, -0.8, ..., contact from x = 0.0 (step 5), also after passing the pole
    expected = np.arange(11) >= 5
    np.testing.assert_array_equal(crashed, expected)


@pytest.mark.unit
def test_crash_mask_through_opening_is_free():
    pos = line([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    far = jnp.array([[10.0, 10.0, 1.5]])
    clearance = obstacle_clearance(pos, far, jnp.zeros(1), radius=0.1)
    assert not jnp.any(crash_mask(pos, clearance, GATES, LIMITS, DRONE_RADIUS))


@pytest.mark.unit
def test_crash_mask_through_frame_and_below_floor():
    far = jnp.array([[10.0, 10.0, 1.5]])
    through_frame = line([-1.0, 0.25, 1.0], [1.0, 0.25, 1.0])
    clearance = obstacle_clearance(through_frame, far, jnp.zeros(1), radius=0.1)
    crashed = crash_mask(through_frame, clearance, GATES, LIMITS, DRONE_RADIUS)
    np.testing.assert_array_equal(crashed, np.arange(11) >= 5)

    below_floor = line([2.0, 2.0, 0.5], [2.0, 2.0, -0.5])
    clearance = obstacle_clearance(below_floor, far, jnp.zeros(1), radius=0.1)
    crashed = crash_mask(below_floor, clearance, GATES, LIMITS, DRONE_RADIUS)
    np.testing.assert_array_equal(crashed, np.arange(11) >= 6)


@pytest.mark.unit
def test_crash_mask_never_flags_initial_state():
    pos = jnp.array([[0.0, 0.0, -1.0], [3.0, 3.0, 1.0]])
    clearance = jnp.ones(2)
    crashed = crash_mask(pos, clearance, GATES, LIMITS, DRONE_RADIUS)
    np.testing.assert_array_equal(crashed, [False, False])


@pytest.mark.unit
def test_crash_mask_ignores_limits_when_starting_outside():
    far = jnp.array([[10.0, 10.0, 1.5]])
    # Starts outside the y limit of 3 m and flies back in: no crash
    back_inside = line([2.0, 3.5, 1.0], [2.0, 2.5, 1.0])
    clearance = obstacle_clearance(back_inside, far, jnp.zeros(1), radius=0.1)
    assert not jnp.any(crash_mask(back_inside, clearance, GATES, LIMITS, DRONE_RADIUS))


@pytest.mark.unit
def test_freeze_after_crash_holds_first_crashed_state():
    X = jnp.arange(5 * 12, dtype=jnp.float32).reshape(5, 12)
    crashed = jnp.array([False, False, True, True, True])
    frozen = freeze_after_crash(X, crashed)
    np.testing.assert_array_equal(frozen[:2], X[:2])
    np.testing.assert_array_equal(frozen[2:], np.tile(X[2], (3, 1)))


@pytest.mark.unit
def test_collision_cost_peaks_at_contact_and_counts_crashed_steps():
    weights = {"w_obs": 2.0, "obstacle_sigma": 0.1, "w_crash": 100.0}
    far = collision_cost(jnp.array([10.0]), jnp.array([False]), weights)
    contact = collision_cost(jnp.array([0.0]), jnp.array([False]), weights)
    inside = collision_cost(jnp.array([-0.05]), jnp.array([False]), weights)
    np.testing.assert_allclose(far, 0.0, atol=1e-6)
    np.testing.assert_allclose(contact, 2.0)
    np.testing.assert_allclose(inside, 2.0)
    crashed = collision_cost(jnp.full(3, 10.0), jnp.array([False, True, True]), weights)
    np.testing.assert_allclose(crashed, 200.0, atol=1e-6)
    assert set(weights) == set(COLLISION_WEIGHTS)


@pytest.mark.unit
def test_obstacle_position_uncertainty():
    kwargs = {"minval": [-0.15, -0.15, -0.05], "maxval": [0.15, 0.1, 0.05]}
    config = SimpleNamespace(env={"randomizations": {"obstacle_pos": {"kwargs": kwargs}}})
    assert obstacle_position_uncertainty(config) == pytest.approx(0.15 * np.sqrt(2))
    assert obstacle_position_uncertainty(SimpleNamespace(env={})) == 0.0


def controller_after_replan(starts_at_drone: bool, vel: list[float]) -> MPPIMPCC:
    """MPPIMPCC state right after switching to a path along the x-axis that starts at x = 1."""
    old = DensePathReference(np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]), version=1)
    new = DensePathReference(np.array([[1.0, 0.0, 1.0], [5.0, 0.0, 1.0]]), version=2)
    mpcc = MPPIMPCC.__new__(MPPIMPCC)
    mpcc._planner = SimpleNamespace(starts_at_drone=starts_at_drone)
    mpcc._objective = ContouringObjective()
    mpcc._carry_window = None
    mpcc._path = path_from_reference(old)
    mpcc._theta_pred = 1.5
    mpcc._v_theta_pred = 0.7
    mpcc._reference = new
    mpcc._replace_path({"vel": np.array(vel)})
    return mpcc


@pytest.mark.unit
def test_replace_path_restarts_progress_when_path_starts_at_drone():
    mpcc = controller_after_replan(True, [3.0, 4.0, 0.0])
    assert mpcc._theta_pred == 0.0
    assert mpcc._v_theta_pred == pytest.approx(3.0, abs=1e-4)
    assert mpcc._path_version == 2
    # Flying against the path gives zero progress speed, and the speed is capped
    assert controller_after_replan(True, [-2.0, 0.0, 0.0])._v_theta_pred == 0.0
    assert controller_after_replan(True, [50.0, 0.0, 0.0])._v_theta_pred == V_THETA_MAX


@pytest.mark.unit
def test_replace_path_projects_progress_otherwise():
    mpcc = controller_after_replan(False, [3.0, 0.0, 0.0])
    # The old point at x = 1.5 lies 0.5 m along the new path, the progress speed is kept
    assert mpcc._theta_pred == pytest.approx(0.5, abs=1e-3)
    assert mpcc._v_theta_pred == 0.7


def contouring_objective(path: ArcLengthPath) -> ContouringObjective:
    """MPCC objective on the given path."""
    objective = ContouringObjective()
    objective.on_new_path(path)
    return objective


def corridor_objective(path: ArcLengthPath) -> CorridorObjective:
    """MPCC++ objective on the given path, with a gate crossing 2 m along it."""
    planner = SimpleNamespace(gate_progress=np.array([2.0]))
    config = SimpleNamespace(
        sim=SimpleNamespace(drone="cf21B_500"), env=SimpleNamespace(sensor_range=0.7)
    )
    objective = CorridorObjective(config, planner)
    objective.on_new_path(path)
    return objective


@pytest.mark.unit
@pytest.mark.parametrize("make_objective", [contouring_objective, corridor_objective])
def test_mppi_mpcc_update_runs_with_rollout_model(make_objective: Callable):
    horizon, n_samples, dt = 5, 32, 0.02
    params = load_params(Dynamics.so_rpy_rotor_drag, "cf21B_500")
    hover_thrust = (params["mass"] * -params["gravity_vec"][-1] - params["acc_coef"]) / params[
        "cmd_f_coef"
    ]
    path = path_from_reference(
        DensePathReference(np.array([[-2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]), version=1)
    )
    # Hovering at the start of the path, with the thrust state at hover thrust
    x0 = jnp.zeros(13).at[:3].set(jnp.array([-2.0, 0.0, 1.0])).at[12].set(hover_thrust)
    U_nom = jnp.tile(jnp.array([0.0, 0.0, 0.0, hover_thrust]), (horizon, 1))
    objective = make_objective(path)

    out = mppi_mpcc_update(
        key=jax.random.key(0),
        x0=x0,
        theta0=jnp.float32(0.0),
        v_theta0=jnp.float32(0.0),
        U_nom=U_nom,
        dv_theta_nom=jnp.zeros(horizon),
        u_prev=U_nom[0],
        path=path,
        speed_limit=objective.speed_limit,
        gates=GATES,
        gates_revealed=jnp.array([True]),
        obstacles_pos=jnp.array([[0.0, 2.0, 1.5]]),
        obstacles_inflation=jnp.zeros(1),
        pos_limits=LIMITS,
        drone_radius=DRONE_RADIUS,
        gate_margin=0.0,
        obstacle_radius=0.015,
        cost_fn=objective.cost_fn,
        cost_inputs=objective.cost_inputs(
            {"gates_pos": np.asarray(GATES[:, :3]), "gates_visited": np.array([True])}
        ),
        collision_weights=COLLISION_WEIGHTS,
        limits=objective.limits,
        params=params,
        dt=dt,
        disc=discretize(params, np.full(horizon, dt)),
        n_samples=n_samples,
        sigma=jnp.array([0.3, 0.3, 0.0, 0.1]),
        sigma_dv_theta=2.0,
        target_ess=0.05,
    )
    U_new, dv_theta_new, theta_plan, _, _, _, _, vis_pos, _, mean_path = out

    assert U_new.shape == (horizon, 4)
    assert dv_theta_new.shape == (horizon,)
    assert vis_pos.shape[1:] == (horizon + 1, 3)
    assert mean_path.shape == (horizon + 1, 3)
    assert np.all(np.isfinite(U_new)) and np.all(np.isfinite(mean_path))
    assert np.all(np.diff(theta_plan) >= 0)
    # Starting from hover, the drone stays close to its altitude within 0.1 s
    np.testing.assert_allclose(mean_path[:, 2], 1.0, atol=0.05)
