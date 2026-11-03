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
