from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.mpcc.corridor import (
    tunnel_barrier,
    tunnel_corners,
    tunnel_frame,
    tunnel_half_width_profile,
)
from lsy_drone_racing.control.mpcc.cost_pp import (
    MPCC_PP_WEIGHTS,
    mpcc_pp_cost,
    progress_speed_weight,
)
from lsy_drone_racing.control.mpcc.path import ArcLengthPath
from lsy_drone_racing.control.planner.centerline import CenterlinePlanner
from lsy_drone_racing.control.utils.utils import drone_collision_radius, obstacle_radius

CONFIG = SimpleNamespace(sim=SimpleNamespace(drone="cf21B_500"))
POLE_CLEARANCE = drone_collision_radius("cf21B_500") + obstacle_radius()
FAR_POLE = [10.0, 10.0, 1.55]


def straight_path(length: float = 10.0, n_points: int = 1001) -> ArcLengthPath:
    """Path along the x-axis at 1 m height."""
    positions = np.zeros((n_points, 3))
    positions[:, 0] = np.linspace(0.0, length, n_points)
    positions[:, 2] = 1.0
    tangents = np.tile([1.0, 0.0, 0.0], (n_points, 1))
    return ArcLengthPath(jnp.asarray(length), jnp.asarray(positions), jnp.asarray(tangents))


def track_obs(
    gate_1_pos: list[float] | None = None,
    visited: list[bool] | None = None,
    pole_pos: list[float] | None = None,
    pole_visited: bool = False,
) -> dict:
    """Two gates: one facing +x, one facing +y passed backwards (towards -y), and one pole."""
    gates_pos = np.array([[1.0, 0.0, 1.0], gate_1_pos or [3.0, 2.0, 1.2]])
    gates_quat = R.from_euler("z", [[0.0], [np.pi / 2]]).as_quat()
    return {
        "pos": np.array([-1.0, 0.0, 0.5]),
        "gates_pos": gates_pos,
        "gates_quat": gates_quat,
        "gates_visited": np.array(visited or [False, False]),
        "gate_sequence": np.array([0, 1]),
        "gate_sequence_direction": np.array([1, -1]),
        "obstacles_pos": np.array([pole_pos or FAR_POLE]),
        "obstacles_visited": np.array([pole_visited]),
    }


def min_pole_distance(planner: CenterlinePlanner, pole_pos: list[float]) -> float:
    """Smallest horizontal distance of the centerline to the pole's axis."""
    reference = planner.plan(track_obs(pole_pos=pole_pos))
    points = reference.path(np.arange(0.0, reference.length, 0.005))
    return float(np.min(np.linalg.norm(points[:, :2] - np.array(pole_pos[:2]), axis=1)))


@pytest.mark.unit
def test_centerline_passes_gate_centers_along_signed_normals():
    planner = CenterlinePlanner(track_obs(), CONFIG)
    reference = planner.plan(track_obs())
    obs = track_obs()
    np.testing.assert_allclose(reference.path(planner.gate_progress), obs["gates_pos"], atol=2e-3)

    tangents = reference.path(planner.gate_progress, 1)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    np.testing.assert_allclose(tangents, [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], atol=1e-2)
    # The path starts at the drone and runs past the last gate
    np.testing.assert_allclose(reference.path(0.0), obs["pos"], atol=1e-6)
    assert reference.length > planner.gate_progress[-1] + 0.9


@pytest.mark.unit
def test_centerline_rebuilds_on_revealed_gate():
    planner = CenterlinePlanner(track_obs(), CONFIG)
    version = planner.plan(track_obs())._version
    # Nothing revealed: same path
    assert planner.plan(track_obs())._version == version

    moved = track_obs(gate_1_pos=[3.1, 2.0, 1.1], visited=[False, True])
    reference = planner.plan(moved)
    assert reference._version != version
    np.testing.assert_allclose(reference.path(planner.gate_progress[1]), [3.1, 2.0, 1.1], atol=2e-3)
    # The start stays fixed
    np.testing.assert_allclose(reference.path(0.0), track_obs()["pos"], atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("pole_pos", [[0.0, 0.0, 1.55], [0.0, 0.05, 1.55], [0.5, -0.03, 1.55]])
def test_centerline_bends_around_poles(pole_pos: list[float]):
    # The straight centerline to the first gate runs along y = 0, through or next to the pole
    planner = CenterlinePlanner(track_obs(pole_pos=pole_pos), CONFIG)
    assert min_pole_distance(planner, pole_pos) >= POLE_CLEARANCE + 0.04
    # It still passes the gates' centers
    reference = planner.plan(track_obs(pole_pos=pole_pos))
    np.testing.assert_allclose(
        reference.path(planner.gate_progress), track_obs()["gates_pos"], atol=2e-3
    )


@pytest.mark.unit
def test_centerline_rebuilds_on_revealed_pole():
    planner = CenterlinePlanner(track_obs(), CONFIG)
    version = planner.plan(track_obs())._version
    revealed = track_obs(pole_pos=[0.0, 0.0, 1.55], pole_visited=True)
    reference = planner.plan(revealed)
    assert reference._version != version
    points = reference.path(np.arange(0.0, reference.length, 0.005))
    assert np.min(np.linalg.norm(points[:, :2], axis=1)) >= POLE_CLEARANCE + 0.04


@pytest.mark.unit
@pytest.mark.parametrize("yaw", [0.0, 0.7, np.pi / 2, -2.5])
def test_tunnel_frame_matches_upright_gate_axes(yaw: float):
    gate_axes = R.from_euler("z", yaw).as_matrix()
    n, b = tunnel_frame(jnp.asarray(gate_axes[:, 0]))
    # n is the gate's horizontal axis (local y), b its vertical axis (local z), up to sign
    np.testing.assert_allclose(np.abs(np.dot(n, gate_axes[:, 1])), 1.0, atol=1e-6)
    np.testing.assert_allclose(b, gate_axes[:, 2], atol=1e-6)


@pytest.mark.unit
def test_tunnel_frame_is_orthonormal_with_horizontal_n():
    t = jnp.array([[0.6, 0.0, 0.8], [0.3, -0.4, 0.2], [-1.0, 1.0, -0.5]])
    t = t / jnp.linalg.norm(t, axis=-1, keepdims=True)
    n, b = tunnel_frame(t)
    np.testing.assert_allclose(n[:, 2], 0.0, atol=1e-6)
    for u, v in ((n, b), (n, t), (b, t)):
        np.testing.assert_allclose(jnp.sum(u * v, axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(b, axis=-1), 1.0, atol=1e-6)
    assert np.all(b[:, 2] > 0)


@pytest.mark.unit
def test_half_width_narrow_at_gates_and_wide_between():
    path = straight_path()
    width = tunnel_half_width_profile(path, np.array([2.0, 7.0]), 0.1, 1.0, 0.35, 0.05)
    s = np.linspace(0.0, 10.0, 1001)
    near_gate = np.abs(s - 2.0) <= 0.1
    assert np.all(width[near_gate] <= 0.1 + 0.01 * 0.9)
    np.testing.assert_allclose(width[np.argmin(np.abs(s - 4.5))], 1.0, atol=1e-3)
    np.testing.assert_allclose(width[np.argmin(np.abs(s - 2.35))], 0.55, atol=1e-3)
    # Symmetric around the gate
    np.testing.assert_allclose(width[150], width[250], atol=1e-6)


@pytest.mark.unit
def test_tunnel_barrier_values():
    alpha, w = 100.0, jnp.float32(0.1)
    zero = jnp.float32(0.0)
    assert tunnel_barrier(zero, zero, w, alpha) < 1e-3
    np.testing.assert_allclose(tunnel_barrier(w, zero, w, alpha), np.log(2.0), atol=1e-3)
    np.testing.assert_allclose(tunnel_barrier(zero, -0.2, w, alpha), 10.0, atol=1e-3)


@pytest.mark.unit
def test_tunnel_corners_span_square():
    positions = jnp.array([[0.0, 0.0, 1.0]])
    corners = tunnel_corners(positions, jnp.array([[1.0, 0.0, 0.0]]), jnp.array([0.2]))
    expected = [[0.2, 1.2], [-0.2, 1.2], [-0.2, 0.8], [0.2, 0.8]]
    np.testing.assert_allclose(np.abs(corners[:, 0, 1]), 0.2, atol=1e-6)
    np.testing.assert_allclose(
        np.sort(corners[:, 0, 1:], axis=0), np.sort(np.array(expected), axis=0), atol=1e-6
    )


def cost_inputs(half_width: float = 1.0, **weights: float) -> dict:
    """MPCC++ weights with overrides and a constant tunnel half-width on `straight_path`.

    The only gate is revealed and far away, so there is no slowdown.
    """
    return {
        "weights": {**MPCC_PP_WEIGHTS, **weights},
        "half_width": jnp.full(1001, half_width),
        "gates_pos": jnp.array([[100.0, 0.0, 1.0]]),
        "gates_revealed": jnp.array([True]),
        "slow_radius": 1.0,
    }


def rollout(offset: list[float], v_theta: float = 0.0, horizon: int = 2) -> dict:
    """Hovering rollout at x = 1 with the given offset from the path, at fixed progress speed."""
    X = jnp.zeros((horizon + 1, 13)).at[:, :3].set(jnp.array([1.0, 0.0, 1.0]) + jnp.array(offset))
    return {
        "X": X,
        "U": jnp.zeros((horizon, 4)),
        "u_prev": jnp.zeros(4),
        "theta": jnp.ones(horizon + 1),
        "v_theta": jnp.full(horizon + 1, v_theta),
        "dv_theta": jnp.zeros(horizon),
        "path": straight_path(),
    }


@pytest.mark.unit
def test_mpcc_pp_cost_splits_contour_weights():
    inputs = cost_inputs(q_contour_h=2.0, q_contour_v=5.0, w_tunnel=0.0)
    np.testing.assert_allclose(mpcc_pp_cost(**rollout([0.0, 0.0, 0.0]), cost_inputs=inputs), 0.0)
    horizontal = mpcc_pp_cost(**rollout([0.0, 0.1, 0.0]), cost_inputs=inputs)
    vertical = mpcc_pp_cost(**rollout([0.0, 0.0, 0.1]), cost_inputs=inputs)
    np.testing.assert_allclose(horizontal, 3 * 2.0 * 0.01, rtol=1e-5)
    np.testing.assert_allclose(vertical, 3 * 5.0 * 0.01, rtol=1e-5)


@pytest.mark.unit
def test_mpcc_pp_cost_preferred_progress_speed():
    inputs = cost_inputs(r_v_theta=2.0, mu=40.0, w_tunnel=0.0)
    costs = [mpcc_pp_cost(**rollout([0.0] * 3, v), cost_inputs=inputs) for v in (9.0, 10.0, 11.0)]
    assert costs[1] < costs[0] and costs[1] < costs[2]


@pytest.mark.unit
def test_mpcc_pp_cost_penalises_leaving_tunnel():
    inputs = cost_inputs(half_width=0.1, q_contour_h=0.0, q_contour_v=0.0)
    inside = mpcc_pp_cost(**rollout([0.0, 0.05, 0.0]), cost_inputs=inputs)
    outside = mpcc_pp_cost(**rollout([0.0, 0.15, 0.0]), cost_inputs=inputs)
    assert inside < 0.1 * MPCC_PP_WEIGHTS["w_tunnel"]
    assert outside > 10 * MPCC_PP_WEIGHTS["w_tunnel"]


@pytest.mark.unit
def test_mpcc_pp_cost_jit_and_vmap():
    inputs = cost_inputs()
    batch = [rollout([0.0, y, 0.0]) for y in (0.0, 0.1, 0.2)]
    stacked = {k: jnp.stack([b[k] for b in batch]) for k in ("X", "U", "theta", "v_theta")}
    batched_cost = jax.jit(
        jax.vmap(mpcc_pp_cost, in_axes=(0, 0, None, 0, 0, None, None, None)), static_argnums=()
    )
    costs = batched_cost(
        stacked["X"],
        stacked["U"],
        jnp.zeros(4),
        stacked["theta"],
        stacked["v_theta"],
        jnp.zeros(2),
        straight_path(),
        inputs,
    )
    expected = [mpcc_pp_cost(**b, cost_inputs=inputs) for b in batch]
    np.testing.assert_allclose(costs, expected, rtol=1e-5)


@pytest.mark.unit
def test_progress_speed_weight_rises_near_unrevealed_gates():
    weights = {**MPCC_PP_WEIGHTS, "r_v_theta": 2.0, "r_v_theta_unrevealed": 8.0}
    gates = jnp.array([[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]])
    path_pos = jnp.array([[0.5, 0.0, 3.0], [2.5, 0.0, 1.0], [5.5, 0.0, 1.0]])
    unrevealed = progress_speed_weight(path_pos, gates, jnp.array([False, False]), 1.0, weights)
    # Within the slow radius (horizontally, the height does not count) and far from both gates
    np.testing.assert_allclose(unrevealed, [10.0, 2.0, 10.0], atol=0.1)
    revealed = progress_speed_weight(path_pos, gates, jnp.array([True, False]), 1.0, weights)
    np.testing.assert_allclose(revealed, [2.0, 2.0, 10.0], atol=0.1)
    # Halfway up at the slow radius
    at_radius = progress_speed_weight(
        jnp.array([1.0, 0.0, 1.0]), gates, ~jnp.zeros(2, bool), 1.0, weights
    )
    np.testing.assert_allclose(at_radius, 2.0, atol=1e-3)
    halfway = progress_speed_weight(
        jnp.array([1.0, 0.0, 1.0]), gates, jnp.zeros(2, bool), 1.0, weights
    )
    np.testing.assert_allclose(halfway, 6.0, atol=1e-3)
