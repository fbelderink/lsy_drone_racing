from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.mpcc.mppi_mpcc import (
    COLLISION_WEIGHTS,
    MPPIMPCC,
    V_THETA_MAX,
    collision_cost,
    crash_mask,
    freeze_after_crash,
)
from lsy_drone_racing.control.mpcc.path import path_from_reference
from lsy_drone_racing.control.planner.pmm import DensePathReference
from lsy_drone_racing.control.utils.collision import (
    gate_contact,
    obstacle_clearance,
    obstacle_position_uncertainty,
)

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
