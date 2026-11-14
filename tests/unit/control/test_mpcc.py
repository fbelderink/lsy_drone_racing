import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.mpcc.cost import (
    contour_weight,
    contouring_errors,
    mpcc_cost,
    progress_step,
)
from lsy_drone_racing.control.mpcc.path import (
    ArcLengthPath,
    path_from_reference,
    path_lookup,
    profile_lookup,
    speed_profile,
)
from lsy_drone_racing.control.planner.reference import Reference

WEIGHTS = {
    "q_nom": 2.0,
    "q_wp": 10.0,
    "gate_sigma": 0.5,
    "q_lag": 3.0,
    "q_rates": 5.0,
    "r_du": jnp.array([7.0, 7.0, 7.0, 11.0]),
    "r_dv_theta": 13.0,
    "mu": 17.0,
}
FAR_GATE = jnp.array([[100.0, 100.0, 100.0]])


def straight_path(length: float = 10.0, n_points: int = 101) -> ArcLengthPath:
    """Path along the x-axis."""
    positions = np.zeros((n_points, 3))
    positions[:, 0] = np.linspace(0.0, length, n_points)
    tangents = np.tile([1.0, 0.0, 0.0], (n_points, 1))
    return ArcLengthPath(jnp.asarray(length), jnp.asarray(positions), jnp.asarray(tangents))


def rollout_inputs(horizon: int = 2) -> dict:
    """Rollout on the path at the origin with all costs zero."""
    return {
        "X": jnp.zeros((horizon + 1, 12)),
        "U": jnp.zeros((horizon, 4)),
        "u_prev": jnp.zeros(4),
        "theta": jnp.zeros(horizon + 1),
        "v_theta": jnp.zeros(horizon + 1),
        "dv_theta": jnp.zeros(horizon),
        "path": straight_path(),
        "gates_pos": FAR_GATE,
        "weights": WEIGHTS,
    }


@pytest.mark.unit
def test_contouring_errors_straight_line():
    contour, lag = contouring_errors(
        jnp.array([2.0, 3.0, 4.0]), jnp.array([1.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(contour, [0.0, 3.0, 4.0])
    np.testing.assert_allclose(lag, 1.0)


@pytest.mark.unit
def test_contouring_errors_on_circle():
    # On a unit circle, arc length equals the angle and a radial offset is purely a contour error
    angles = np.linspace(0.0, 2 * np.pi, 2048)
    radial = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=-1)
    tangents = np.stack((-np.sin(angles), np.cos(angles), np.zeros_like(angles)), axis=-1)
    path = ArcLengthPath(jnp.asarray(2 * np.pi), jnp.asarray(radial), jnp.asarray(tangents))

    theta = 1.0
    expected_radial = np.array([np.cos(theta), np.sin(theta), 0.0])
    path_pos, tangent = path_lookup(path, jnp.asarray(theta))
    contour, lag = contouring_errors(jnp.asarray(1.5 * expected_radial), path_pos, tangent)

    np.testing.assert_allclose(contour, 0.5 * expected_radial, atol=1e-3)
    np.testing.assert_allclose(lag, 0.0, atol=1e-3)


@pytest.mark.unit
def test_contouring_errors_decomposition():
    key_pos, key_path, key_tangent = jax.random.split(jax.random.key(0), 3)
    pos = jax.random.normal(key_pos, (20, 3))
    path_pos = jax.random.normal(key_path, (20, 3))
    tangent = jax.random.normal(key_tangent, (20, 3))
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)

    contour, lag = contouring_errors(pos, path_pos, tangent)

    np.testing.assert_allclose(jnp.sum(contour * tangent, axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(contour + lag[:, None] * tangent, pos - path_pos, atol=1e-5)


@pytest.mark.unit
def test_path_from_reference_matches_reference():
    reference = Reference.default_reference({"pos": np.array([-1.5, 1.0, 0.1])})
    path = path_from_reference(reference)

    theta = np.linspace(0.0, reference.length, 57)
    positions, tangents = path_lookup(path, jnp.asarray(theta, dtype=jnp.float32))

    np.testing.assert_allclose(positions, reference.path(theta), atol=1e-3)
    np.testing.assert_allclose(jnp.linalg.norm(tangents, axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.norm(path.tangents, axis=-1), 1.0, atol=1e-5)
    expected_tangents = reference.path(theta, 1)
    expected_tangents /= np.linalg.norm(expected_tangents, axis=-1, keepdims=True)
    np.testing.assert_allclose(tangents, expected_tangents, atol=1e-2)


@pytest.mark.unit
def test_path_lookup_clamps_progress():
    path = straight_path()
    positions, _ = path_lookup(path, jnp.array([-1.0, 0.0, 10.0, 11.0]))
    np.testing.assert_allclose(positions[:, 0], [0.0, 0.0, 10.0, 10.0])


@pytest.mark.unit
def test_path_shape_is_fixed_after_replan():
    reference = Reference.default_reference({"pos": np.array([-1.5, 1.0, 0.1])})
    before = path_from_reference(reference)
    n_points_before = len(reference.path.x)
    length_before = reference.length

    s = np.array([0.5 * reference.length])
    reference.update(reference.path(s) + np.array([0.0, 0.0, 0.3]), s, min_spacing=0.0)
    after = path_from_reference(reference)

    # The replan changes the reference's size, but not the size of the resampled path
    assert len(reference.path.x) != n_points_before
    assert reference.length != length_before
    assert after.positions.shape == before.positions.shape
    assert after.tangents.shape == before.tangents.shape
    np.testing.assert_allclose(after.length, reference.length, rtol=1e-6)
    theta = np.linspace(0.0, reference.length, 57)
    positions, _ = path_lookup(after, jnp.asarray(theta, dtype=jnp.float32))
    np.testing.assert_allclose(positions, reference.path(theta), atol=1e-3)


@pytest.mark.unit
def test_contour_weight_peak_and_floor():
    gates = jnp.array([[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]])
    weight = contour_weight(jnp.array([[0.0, 0.0, 1.0], [50.0, 0.0, 1.0]]), gates, WEIGHTS)
    np.testing.assert_allclose(weight, [WEIGHTS["q_nom"] + WEIGHTS["q_wp"], WEIGHTS["q_nom"]])


@pytest.mark.unit
def test_contour_weight_at_one_sigma():
    weight = contour_weight(jnp.array([0.5, 0.0, 0.0]), jnp.zeros((1, 3)), WEIGHTS)
    np.testing.assert_allclose(weight, 2.0 + 10.0 * np.exp(-0.5), rtol=1e-6)


@pytest.mark.unit
def test_progress_step_integrates():
    theta, v_theta = progress_step(1.0, 2.0, 3.0, 0.1, 10.0, 5.0)
    np.testing.assert_allclose(theta, 1.2)
    np.testing.assert_allclose(v_theta, 2.3)


@pytest.mark.unit
@pytest.mark.parametrize(("dv_theta", "expected"), [(-100.0, 0.0), (100.0, 5.0)])
def test_progress_step_limits_speed(dv_theta: float, expected: float):
    _, v_theta = progress_step(1.0, 2.0, dv_theta, 0.1, 10.0, 5.0)
    np.testing.assert_allclose(v_theta, expected)


@pytest.mark.unit
def test_progress_step_stops_at_path_end():
    theta, v_theta = progress_step(9.9, 2.0, 1.0, 0.1, 10.0, 5.0)
    np.testing.assert_allclose(theta, 10.0)
    np.testing.assert_allclose(v_theta, 0.0)


@pytest.mark.unit
def test_cost_hand_computed():
    inputs = rollout_inputs(horizon=1)
    # At theta = 1 the path point is (1, 0, 0): lag error 0.5, contour error 0.2 in y
    inputs["X"] = jnp.zeros((2, 12)).at[1, :3].set(jnp.array([1.5, 0.2, 0.0]))
    inputs["X"] = inputs["X"].at[1, 9:12].set(jnp.array([0.0, 0.0, 2.0]))
    inputs["theta"] = jnp.array([0.0, 1.0])
    inputs["v_theta"] = jnp.array([1.0, 2.0])
    inputs["dv_theta"] = jnp.array([0.5])
    inputs["U"] = jnp.array([[0.1, 0.0, 0.0, 0.2]])

    contour = 2.0 * 0.2**2
    lag = 3.0 * 0.5**2
    rates = 5.0 * 2.0**2
    commands = 7.0 * 0.1**2 + 11.0 * 0.2**2
    progress_change = 13.0 * 0.5**2
    reward = 17.0 * (1.0 + 2.0)
    expected = contour + lag + rates + commands + progress_change - reward

    np.testing.assert_allclose(mpcc_cost(**inputs), expected, rtol=1e-6)


@pytest.mark.unit
def test_cost_is_zero_on_path_without_motion():
    np.testing.assert_allclose(mpcc_cost(**rollout_inputs()), 0.0)


@pytest.mark.unit
def test_cost_ordering():
    base = rollout_inputs()
    base_cost = mpcc_cost(**base)

    assert mpcc_cost(**{**base, "v_theta": jnp.ones(3)}) < base_cost
    assert mpcc_cost(**{**base, "X": base["X"].at[:, 1].set(0.1)}) > base_cost
    assert mpcc_cost(**{**base, "X": base["X"].at[:, 0].set(0.1)}) > base_cost
    assert mpcc_cost(**{**base, "U": base["U"].at[:, 3].set(0.1)}) > base_cost


@pytest.mark.unit
def test_contour_error_costs_more_near_gate():
    base = rollout_inputs()
    off_path = base["X"].at[:, 1].set(0.1)
    far = mpcc_cost(**{**base, "X": off_path})
    near = mpcc_cost(**{**base, "X": off_path, "gates_pos": jnp.zeros((1, 3))})
    assert near > far


@pytest.mark.unit
def test_cost_works_with_jit_and_vmap():
    inputs = rollout_inputs()
    n_samples = 8
    batched_cost = jax.jit(jax.vmap(mpcc_cost, in_axes=(0, 0, None, 0, 0, 0, None, None, None)))
    costs = batched_cost(
        jnp.tile(inputs["X"], (n_samples, 1, 1)),
        jnp.tile(inputs["U"], (n_samples, 1, 1)),
        inputs["u_prev"],
        jnp.tile(inputs["theta"], (n_samples, 1)),
        jnp.tile(inputs["v_theta"], (n_samples, 1)),
        jnp.tile(inputs["dv_theta"], (n_samples, 1)),
        inputs["path"],
        inputs["gates_pos"],
        inputs["weights"],
    )
    assert costs.shape == (n_samples,)
    assert jnp.all(jnp.isfinite(costs))


def circle_path(radius: float, n_points: int = 2048) -> ArcLengthPath:
    """Half circle in the x-y plane."""
    angles = np.linspace(0.0, np.pi, n_points)
    positions = radius * np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=1)
    tangents = np.stack((-np.sin(angles), np.cos(angles), np.zeros_like(angles)), axis=1)
    return ArcLengthPath(jnp.asarray(np.pi * radius), jnp.asarray(positions), jnp.asarray(tangents))


@pytest.mark.unit
@pytest.mark.parametrize("radius", [0.5, 2.0])
def test_speed_profile_on_a_circle(radius: float):
    profile = np.asarray(speed_profile(circle_path(radius), 12.0, 6.0, 100.0, 0.05))
    # The tangent difference measures the chord, so allow a small error
    np.testing.assert_allclose(profile[:1800], np.sqrt(12.0 * radius), rtol=0.01)


@pytest.mark.unit
def test_speed_profile_brakes_before_a_turn():
    # 5 m straight along x, then a sharp 90 degree corner and 5 m along y
    straight = np.linspace(0.0, 5.0, 1024, endpoint=False)
    zeros = np.zeros_like(straight)
    first_leg = np.stack((straight, zeros, zeros), axis=1)
    second_leg = np.stack((zeros + 5.0, straight, zeros), axis=1)
    positions = np.concatenate((first_leg, second_leg))
    tangents = np.concatenate((np.tile([1.0, 0, 0], (1024, 1)), np.tile([0.0, 1, 0], (1024, 1))))
    path = ArcLengthPath(jnp.asarray(10.0), jnp.asarray(positions), jnp.asarray(tangents))
    profile = np.asarray(speed_profile(path, 12.0, 6.0, 6.0, 0.05))
    corner = 1023
    assert profile[corner] < 3.0  # Slow in the corner
    assert profile[0] == pytest.approx(6.0)  # Capped far before it
    # The speed rises from the corner backwards no faster than braking at 6 m/s^2 allows
    ds = 10.0 / 2047
    before = np.arange(800, corner)
    reachable = profile[corner] ** 2 + 2 * 6.0 * ds * (corner - before)
    assert np.all(profile[before] ** 2 <= reachable + 1e-3)


@pytest.mark.unit
def test_profile_lookup_interpolates_and_clamps():
    profile = jnp.array([0.0, 1.0, 2.0])
    values = profile_lookup(profile, jnp.asarray(2.0), jnp.array([0.5, 1.5, 5.0]))
    np.testing.assert_allclose(values, [0.5, 1.5, 2.0])
