"""Collision geometry of the race track, shared by planners and controllers.

Gates are square frames on a stand, obstacles are vertical poles, and the drone is a sphere. The
sizes come from `envs/assets/gate.xml` and `obstacle.xml`. The geometry functions use `jax.numpy`
when given JAX arrays, e.g. in jitted code, and plain numpy otherwise, so planners calling them
with paths of changing length do not trigger JAX compilations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from jax import Array

# Gate geometry in meters, from `envs/assets/gate.xml`
GATE_INNER_HALF_WIDTH = 0.2  # Half-width of the gate opening
GATE_OUTER_HALF_WIDTH = 0.36  # Half-width of the frame's outer edge
GATE_HALF_DEPTH = 0.01  # Half-thickness of the frame along the gate's normal
GATE_STAND_HALF_WIDTH = 0.05  # Half-width of the square stand below the frame


def array_module(*arrays: Array) -> ModuleType:
    """Return `jax.numpy` if any of the arrays is a JAX array or tracer, else `numpy`."""
    return jnp if any(isinstance(a, jax.Array) for a in arrays) else np


def obstacle_clearance(pos: Array, obstacles_pos: Array, inflation: Array, radius: float) -> Array:
    """Horizontal clearance between the drone and the surface of the nearest obstacle.

    Args:
        pos: Positions with shape (n, 3).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3). Obstacles are vertical
            poles, so only the x and y coordinates are used.
        inflation: Extra radius in meters per obstacle with shape (n_obstacles,), covering the
            position uncertainty of obstacles whose true position is not revealed yet.
        radius: Sum of the drone's and the pole's collision radius in meters.

    Returns:
        The clearance in meters with shape (n,), negative when in contact.
    """
    xp = array_module(pos, obstacles_pos, inflation)
    xy_distance = xp.linalg.norm(pos[:, None, :2] - obstacles_pos[None, :, :2], axis=-1)
    return xp.min(xy_distance - radius - inflation[None, :], axis=-1)


def gate_contact(pos: Array, gates: Array, drone_radius: float) -> Array:
    """Check which positions touch a gate's frame or stand.

    The gate's vertical axis is taken as the world z-axis, so its small roll and pitch are ignored.

    Args:
        pos: Positions with shape (n, 3).
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        drone_radius: Collision radius of the drone in meters.

    Returns:
        Contact flags with shape (n,).
    """
    xp = array_module(pos, gates)
    offset = pos[:, None, :] - gates[None, :, :3]
    normal = gates[:, 3:]
    horizontal_axis = xp.cross(normal, xp.array([0.0, 0.0, 1.0]))
    along_normal = xp.abs(xp.sum(offset * normal, axis=-1))
    horizontal = xp.abs(xp.sum(offset * horizontal_axis, axis=-1))
    vertical = offset[..., 2]
    # Frame: a square ring in the gate's plane, checked with the max-norm of the in-plane offset
    in_plane = xp.maximum(horizontal, xp.abs(vertical))
    frame = (
        (along_normal < GATE_HALF_DEPTH + drone_radius)
        & (in_plane > GATE_INNER_HALF_WIDTH - drone_radius)
        & (in_plane < GATE_OUTER_HALF_WIDTH + drone_radius)
    )
    # Stand: a square column from the frame's bottom edge down to the ground
    stand = (
        (along_normal < GATE_STAND_HALF_WIDTH + drone_radius)
        & (horizontal < GATE_STAND_HALF_WIDTH + drone_radius)
        & (vertical < -GATE_OUTER_HALF_WIDTH)
    )
    return xp.any(frame | stand, axis=-1)


def obstacle_position_uncertainty(config: dict) -> float:
    """Largest horizontal offset of an obstacle from its nominal position.

    Args:
        config: The race configuration.

    Returns:
        The offset in meters, 0 if the obstacle positions are not randomized.
    """
    randomization = (config.env.get("randomizations") or {}).get("obstacle_pos")
    if randomization is None:
        return 0.0
    kwargs = randomization["kwargs"]
    bound = np.maximum(np.abs(kwargs["minval"]), np.abs(kwargs["maxval"]))
    return float(np.linalg.norm(bound[:2]))
