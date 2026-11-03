"""Model Predictive Contouring Control cost, after Romero et al., T-RO 2022.

The cost rewards progress along a path while penalising the distance to it. It is adapted to the
collective thrust and attitude interface, using the conventions of `attitude_mppi.py`:

    x = [pos(3), rpy(3), vel(3), drpy(3)]
    u = [roll, pitch, yaw, thrust]

Compared to Eq. 17 of the paper, command differences replace the single rotor thrust differences,
and rpy rates replace the body rates. Progress theta and its speed v_theta are extra states, driven
by the virtual input dv_theta (Eq. 16).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from lsy_drone_racing.control.mpcc.path import path_lookup

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.control.mpcc.path import ArcLengthPath

# Placeholder values until an optimizer uses this cost
MPCC_WEIGHTS = {
    "q_nom": 1.0,  # Contour weight away from gates
    "q_wp": 100.0,  # Additional contour weight at a gate's center
    "gate_sigma": 0.3,  # Width in meters of the Gaussian around each gate
    "q_lag": 100.0,  # Lag weight, high so that theta stays a good estimate of the closest point
    "q_rates": 0.01,  # Weight for rpy rates
    "r_du": jnp.array([1.0, 1.0, 1.0, 1.0]),  # Weights for command changes per channel
    "r_dv_theta": 0.01,  # Weight for changes of the progress speed
    "mu": 1.0,  # Reward for progress speed
}


def contouring_errors(pos: Array, path_pos: Array, tangent: Array) -> tuple[Array, Array]:
    """Split the position error into components normal and tangential to the path.

    Args:
        pos: Positions with shape (..., 3).
        path_pos: Path positions at the current progress with shape (..., 3).
        tangent: Unit path tangents at the current progress with shape (..., 3).

    Returns:
        - The contour error vectors with shape (..., 3).
        - The signed lag errors with shape (...,).
    """
    error = pos - path_pos
    lag = jnp.sum(tangent * error, axis=-1)
    return error - lag[..., None] * tangent, lag


def contour_weight(path_pos: Array, gates_pos: Array, weights: dict) -> Array:
    """Compute the contour weight, which rises on the parts of the path near a gate.

    The Gaussians are peak-normalised, so `q_wp` is directly the additional weight at a gate's
    center. They depend on the path position and not on the drone position, so the weight does not
    change when the drone is off the path. A path that misses a gate's center gets a lower peak, and
    overlapping Gaussians add up.

    Args:
        path_pos: Path positions with shape (..., 3).
        gates_pos: Gate positions with shape (n_gates, 3).
        weights: Cost weights, see `MPCC_WEIGHTS`.

    Returns:
        The contour weights with shape (...,).
    """
    squared_distance = jnp.sum((path_pos[..., None, :] - gates_pos) ** 2, axis=-1)
    gaussians = jnp.exp(-0.5 * squared_distance / weights["gate_sigma"] ** 2)
    return weights["q_nom"] + weights["q_wp"] * jnp.sum(gaussians, axis=-1)


def progress_step(
    theta: Array, v_theta: Array, dv_theta: Array, dt: float, length: Array, v_theta_max: float
) -> tuple[Array, Array]:
    """Advance the progress states by one time step (Eq. 16).

    Args:
        theta: Progress in meters.
        v_theta: Progress speed in m/s.
        dv_theta: Progress acceleration in m/s^2.
        dt: Integration step in seconds.
        length: Path length in meters, at which progress stops.
        v_theta_max: Maximum progress speed in m/s. The minimum is 0, so progress never reverses.

    Returns:
        The next progress and the next progress speed.
    """
    theta_next = jnp.minimum(theta + v_theta * dt, length)
    v_theta_next = jnp.clip(v_theta + dv_theta * dt, 0.0, v_theta_max)
    # No progress speed at the path's end, so nothing beyond it is rewarded
    return theta_next, jnp.where(theta_next >= length, 0.0, v_theta_next)


def mpcc_cost(
    X: Array,
    U: Array,
    u_prev: Array,
    theta: Array,
    v_theta: Array,
    dv_theta: Array,
    path: ArcLengthPath,
    gates_pos: Array,
    weights: dict,
) -> Array:
    """Compute the MPCC cost of a single rollout.

    Args:
        X: State trajectory with shape (H + 1, 12).
        U: Control sequence with shape (H, 4).
        u_prev: Previous control input with shape (4,).
        theta: Progress trajectory with shape (H + 1,).
        v_theta: Progress speed trajectory with shape (H + 1,).
        dv_theta: Progress acceleration sequence with shape (H,).
        path: The path to follow.
        gates_pos: Gate positions with shape (n_gates, 3).
        weights: Cost weights, see `MPCC_WEIGHTS`.

    Returns:
        The scalar cost.
    """
    path_pos, tangent = path_lookup(path, theta)
    contour, lag = contouring_errors(X[:, :3], path_pos, tangent)

    contour_cost = jnp.sum(
        contour_weight(path_pos, gates_pos, weights) * jnp.sum(contour**2, axis=-1)
    )
    lag_cost = weights["q_lag"] * jnp.sum(lag**2)
    rates_cost = weights["q_rates"] * jnp.sum(X[:, 9:12] ** 2)
    progress_reward = weights["mu"] * jnp.sum(v_theta)

    command_change = jnp.diff(jnp.vstack((u_prev, U)), axis=0)
    command_cost = jnp.sum(weights["r_du"] * command_change**2)
    progress_change_cost = weights["r_dv_theta"] * jnp.sum(dv_theta**2)

    return (
        contour_cost + lag_cost + rates_cost + command_cost + progress_change_cost - progress_reward
    )
