"""Prismatic tunnel around the path, after Krinner, Romero et al., "MPCC++", RSS 2024 (Sec. IV-A).

The tunnel is a square cross-section centered on the path, with the same half-width W(theta) to the
sides and up and down. It is wide between gates and narrows to the gate opening at every gate, so
staying inside it keeps the drone clear of the gate frames while it is free to cut corners
elsewhere.

The cross-section's axes are gravity-aligned instead of the paper's Frenet-Serret frame: n is
horizontal and b points up, tilted by the path's climb. Unlike the Frenet-Serret frame, this is
defined on straight parts of the path, which is where the path passes the gates, and its axes
coincide with the axes of an upright gate's opening.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jax import Array
    from numpy.typing import NDArray

    from lsy_drone_racing.control.mpcc.path import ArcLengthPath

WORLD_UP = jnp.array([0.0, 0.0, 1.0])


def tunnel_frame(tangent: Array) -> tuple[Array, Array]:
    """Gravity-aligned axes of the tunnel's cross-section.

    Undefined for vertical tangents, which do not occur on the race tracks.

    Args:
        tangent: Unit path tangents with shape (..., 3).

    Returns:
        - The horizontal axis n = normalize(z x t) with shape (..., 3).
        - The upward axis b = t x n with shape (..., 3).
    """
    n = jnp.cross(WORLD_UP, tangent)
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    return n, jnp.cross(tangent, n)


def tunnel_half_width_profile(
    path: ArcLengthPath,
    gate_progress: NDArray[np.floating],
    w_gate: float,
    w_nominal: float,
    d_mid: float,
    steepness: float,
) -> Array:
    """Half-width of the tunnel at every point of the path's grid.

    The half-width goes from w_gate at a gate to w_nominal away from gates along a sigmoid of the
    arc-length distance d to the nearest gate crossing:

        W = w_gate + (w_nominal - w_gate) * sigmoid((d - d_mid) / steepness)

    It is within 1 % of the change of w_gate for d <= d_mid - 5 * steepness.

    Args:
        path: The path, sampled on a uniform arc-length grid.
        gate_progress: Arc length in meters of every gate crossing with shape (n_gates,).
        w_gate: Half-width in meters at a gate.
        w_nominal: Half-width in meters away from gates.
        d_mid: Distance in meters from a gate at which the half-width is halfway between both.
        steepness: Length in meters over which the half-width changes.

    Returns:
        The half-widths in meters with the shape (N,) of the path's grid.
    """
    s = np.linspace(0.0, float(path.length), path.positions.shape[0])
    distance = np.min(np.abs(s[:, None] - np.asarray(gate_progress)[None, :]), axis=1)
    sigmoid = 1.0 / (1.0 + np.exp(-(distance - d_mid) / steepness))
    return jnp.asarray(w_gate + (w_nominal - w_gate) * sigmoid, dtype=jnp.float32)


def tunnel_barrier(e_n: Array, e_b: Array, half_width: Array, alpha: float) -> Array:
    """Soft barrier on the four walls of the tunnel (Eq. 7 and 14).

    Each wall is a halfspace constraint h >= 0, penalised by log(1 + exp(-alpha * h)), which is
    about 0 well inside, log(2) on the wall and grows linearly with slope alpha outside.

    Args:
        e_n: Position error along the horizontal tunnel axis n in meters with shape (...,).
        e_b: Position error along the upward tunnel axis b in meters with shape (...,).
        half_width: Half-width of the tunnel in meters with shape (...,).
        alpha: Slope of the barrier in 1/m.

    Returns:
        The barrier values summed over the four walls with shape (...,).
    """
    walls = jnp.stack((half_width - e_n, half_width + e_n, half_width - e_b, half_width + e_b))
    return jnp.sum(jax.nn.softplus(-alpha * walls), axis=0)


def tunnel_corners(positions: Array, tangents: Array, half_width: Array) -> Array:
    """Corners of the tunnel's cross-sections, e.g. to draw its four edges.

    Args:
        positions: Path positions with shape (M, 3).
        tangents: Unit path tangents with shape (M, 3).
        half_width: Half-widths in meters with shape (M,).

    Returns:
        The corners with shape (4, M, 3), in the order (+n +b), (-n +b), (-n -b), (+n -b).
    """
    n, b = tunnel_frame(tangents)
    w = half_width[:, None]
    return jnp.stack(
        (
            positions + w * n + w * b,
            positions - w * n + w * b,
            positions - w * n - w * b,
            positions + w * n - w * b,
        )
    )
