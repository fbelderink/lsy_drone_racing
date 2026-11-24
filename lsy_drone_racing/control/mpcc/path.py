"""Fixed-size JAX representation of a planned path, indexed by arc length.

`Reference` changes its number of points on every replan. Resampling it to a constant size keeps
array shapes fixed, so jitted code does not recompile after a replan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
import numpy.linalg as la

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.control.planner.interface import PathReference
    from lsy_drone_racing.control.planner.reference import Reference

N_PATH_POINTS = 2048  # Grid size of the resampled path, about 1 cm spacing on a 20 m path


class ArcLengthPath(NamedTuple):
    """Path sampled on a uniform arc-length grid."""

    length: Array  # Total arc length in meters
    positions: Array  # Positions with shape (N, 3)
    tangents: Array  # Unit tangents with shape (N, 3)


def path_from_reference(reference: Reference) -> ArcLengthPath:
    """Resample a planned reference to a fixed-size path.

    Args:
        reference: The reference to resample.

    Returns:
        The resampled path.
    """
    s = np.linspace(0.0, reference.length, N_PATH_POINTS)
    tangents = reference.path(s, 1)
    tangents = tangents / la.norm(tangents, axis=-1, keepdims=True)
    return ArcLengthPath(
        length=jnp.asarray(reference.length, dtype=jnp.float32),
        positions=jnp.asarray(reference.path(s), dtype=jnp.float32),
        tangents=jnp.asarray(tangents, dtype=jnp.float32),
    )


def path_lookup(path: ArcLengthPath, theta: Array) -> tuple[Array, Array]:
    """Interpolate position and unit tangent at the given progress.

    Args:
        path: The path to evaluate.
        theta: Progress in meters with any shape (...,). Values outside the path are clamped.

    Returns:
        Positions and unit tangents, each with shape (..., 3).
    """
    n_segments = path.positions.shape[0] - 1
    grid_position = jnp.clip(theta, 0.0, path.length) / path.length * n_segments
    i = jnp.clip(jnp.floor(grid_position).astype(jnp.int32), 0, n_segments - 1)
    w = (grid_position - i)[..., None]

    positions = (1 - w) * path.positions[i] + w * path.positions[i + 1]
    tangents = (1 - w) * path.tangents[i] + w * path.tangents[i + 1]
    return positions, tangents / jnp.linalg.norm(tangents, axis=-1, keepdims=True)


def carry_progress(
    old_path: ArcLengthPath,
    theta: float,
    new_reference: PathReference,
    search_range_s: tuple[float, float] | None = None,
) -> float:
    """Convert a progress on a replaced path to the progress on its replacement.

    Arc lengths are not comparable between paths, e.g. a path re-planned from the drone's current
    position starts again at zero. The point at theta on the old path is projected onto the new
    path instead.

    Args:
        old_path: The path before the replan.
        theta: Progress in meters on the old path.
        new_reference: The re-planned path.
        search_range_s: (behind, ahead) window in meters around theta to search on the new path,
            for paths whose arc length changes little near the drone, e.g. paths with the same
            start. None searches the whole new path.

    Returns:
        The progress in meters on the new path.
    """
    old_point = np.asarray(path_lookup(old_path, jnp.float32(theta))[0])
    if search_range_s is None:
        return new_reference.get_progress(old_point)
    return new_reference.get_progress(old_point, prev_progress=theta, search_range_s=search_range_s)


def speed_profile(
    path: ArcLengthPath, a_lat: float, a_brake: float, v_max: float, curvature_ds: float
) -> Array:
    """Highest speed at every path point at which the drone can follow the path's turns.

    The speed is limited to v <= sqrt(a_lat / k) for the curvature k, and a backward pass limits it
    further so that slowing down for a turn ahead never needs more deceleration than a_brake.

    Args:
        path: The path, sampled on a uniform grid.
        a_lat: Sideways acceleration in m/s^2 allowed in turns.
        a_brake: Deceleration in m/s^2 allowed before turns.
        v_max: Speed limit in m/s on straight parts.
        curvature_ds: Arc length in meters over which the curvature is measured.

    Returns:
        The speed limits in m/s with the shape (N,) of the path's grid.
    """
    tangents = np.asarray(path.tangents)
    ds = float(path.length) / (len(tangents) - 1)
    step = max(int(round(curvature_ds / ds)), 1)
    ahead = np.minimum(np.arange(len(tangents)) + step, len(tangents) - 1)
    curvature = la.norm(tangents[ahead] - tangents, axis=1) / (step * ds)
    speed = np.minimum(np.sqrt(a_lat / np.maximum(curvature, 1e-6)), v_max)
    for i in range(len(speed) - 2, -1, -1):
        speed[i] = min(speed[i], np.sqrt(speed[i + 1] ** 2 + 2 * a_brake * ds))
    return jnp.asarray(speed, dtype=jnp.float32)


def profile_lookup(profile: Array, length: Array, theta: Array) -> Array:
    """Interpolate a profile sampled on a path's grid at the given progress.

    Args:
        profile: Values on the path's uniform arc-length grid with shape (N,).
        length: The path's length in meters.
        theta: Progress in meters with any shape (...,). Values outside the path are clamped.

    Returns:
        The interpolated values with the shape of theta.
    """
    n_segments = profile.shape[0] - 1
    grid_position = jnp.clip(theta, 0.0, length) / length * n_segments
    i = jnp.clip(jnp.floor(grid_position).astype(jnp.int32), 0, n_segments - 1)
    w = grid_position - i
    return (1 - w) * profile[i] + w * profile[i + 1]
