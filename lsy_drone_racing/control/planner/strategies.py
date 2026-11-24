"""Collision avoidance for the high level planner's reference path.

The reference is sampled densely and checked against the obstacles seen so far. Where it passes
a pole too closely, a waypoint pushed horizontally away from the pole is inserted as an extra
point into the reference. This is repeated a few times, since a pushed path can collide again.
"""

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray

from lsy_drone_racing.control.planner.reference import ProgressReference
from lsy_drone_racing.control.utils.types import ObservationDict
from lsy_drone_racing.control.utils.utils import drone_collision_radius, obstacle_radius


def check_gate_collision(ref: NDArray, gate_pose: NDArray, clearance: float) -> NDArray:
    """Find the closest points to each gate the reference passes too close to.

    Ignores reference points in the vicinity of gates passes.

    Args:
        ref: (N, 4) progress + points on the reference [s, x, y, z]
        gate_pose: (G, 6) poses of the gates [x, y, z, roll, pitch, yaw]
        clearance: Minimum clearance between every point on the reference and the gates geometry.
    """
    pass


def check_obstacle_collision(
    ref: NDArray, obstacle_pos: NDArray, clearance: float
) -> tuple[NDArray, NDArray]:
    """Find the closest points to each obstacle the reference passes too close to.

    Args:
        ref: (N, 3) points on the reference.
        obstacle_pos: (O, 3) obstacle positions.
        clearance: Minimum horizontal distance to the pole axes.

    Returns:
        dist (N, O) horizontal distance to the pole axes (inf above a pole), and a mask (N, O)
        of the local distance minima within each too-close stretch of the reference.
    """
    d = ref[:, None, :] - obstacle_pos[None, :, :]  # (N, O, 3)
    dist = np.where(d[..., 2] > clearance, np.inf, la.norm(d[..., :2], axis=-1))  # (N, O)

    d_in = np.where(dist <= clearance, dist, np.inf)  # inf outside of the collisions
    prev = np.pad(d_in, ((1, 0), (0, 0)), constant_values=np.inf)[:-1]
    nxt = np.pad(d_in, ((0, 1), (0, 0)), constant_values=np.inf)[1:]
    return dist, (d_in <= prev) & (d_in < nxt)


def push_vectors(
    pos: NDArray,
    tangent: NDArray,
    obstacles: NDArray,
    dist: NDArray,
    mask: NDArray,
    clearance: float,
) -> NDArray:
    """Horizontal push per sample that moves it to `target` distance from every masked obstacle.

    The pushes of all obstacles masked at the same sample are summed. Obstacles on the same side
    add up (conservative), obstacles on opposite sides cancel, which leaves the point in between.

    Args:
        pos: (N, 3) path positions.
        tangent: (N, 3) unit tangents of the path.
        obstacles: (O, 3) obstacle positions.
        dist: (N, O) horizontal distance to the pole axes.
        mask: (N, O) samples to push away from each obstacle.
        clearance: Distance from the pole axis the pushed point should have.

    Returns:
        (N, 2) push in xy, zero where nothing is masked.
    """
    away = pos[:, None, :2] - obstacles[None, :, :2]  # (N, O, 2) pole axis -> path
    norm = la.norm(away, axis=-1, keepdims=True)
    # if the path passes through the axis exactly, push along the horizontal normal of the path
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    normal /= np.maximum(la.norm(normal, axis=-1, keepdims=True), 1e-9)
    away = np.where(norm > 1e-6, away / np.maximum(norm, 1e-9), normal[:, None, :])

    deficit = np.where(mask, clearance - dist, 0.0)  # (N, O) how far to push, 0 if not masked
    return np.sum(deficit[..., None] * away, axis=1)


class CollisionAvoidanceStrategy:
    """Strategy for High Level Planner to avoid collisions in path."""

    def __init__(self, cfg: dict, n_iters: int = 3, n_samples: int = 1000):
        """Initialize Collision Avoidance Strategy."""
        margin = 0.01  # [m] safety margin for collisions
        self._clearance = drone_collision_radius(cfg.sim.drone) + obstacle_radius() + margin
        self.n_iters = n_iters
        self.n_samples = n_samples

    def avoid_collisions(self, obs: ObservationDict, ref: ProgressReference) -> ProgressReference:
        """Push the reference away from obstacles until it is clear or n_iters is reached."""
        for _ in range(self.n_iters):
            new_ref = self.avoid_obstacle_collision(obs, ref)
            new_ref = self.avoid_gate_collision(obs, ref)
            if new_ref is ref:  # nothing collided
                break
            ref = new_ref

        return ref

    def avoid_gate_collision(
        self, obs: ObservationDict, ref: ProgressReference, max_pushes: int = 8
    ) -> ProgressReference:
        """Push the reference away from gates.

        Takes the local distance minima per gate (check_gate_collision).
        Returns ref unchanged if nothing collides.
        """
        return ref

    def avoid_obstacle_collision(
        self, obs: ObservationDict, ref: ProgressReference, max_pushes: int = 8
    ) -> ProgressReference:
        """Push the reference away from obstacles.

        Takes the local distance minima per obstacle (check_obstacle_collision), combines
        obstacles that hit the same sample into one push, and inserts up to max_pushes push
        points. Returns ref unchanged if nothing collides.
        """
        s = np.linspace(0, ref.length, self.n_samples)
        pos, tangent = ref(s, nu=0), ref(s, nu=1)  # (N, 3)
        obstacles = obs["obstacles_pos"]

        # inflate by ds/2: then clear samples guarantee a clear path in between
        ds = s[1] - s[0]
        dist, mask = check_obstacle_collision(pos, obstacles, self._clearance + ds / 2)
        mask &= obs["obstacles_visited"][None, :]

        idx = np.flatnonzero(mask.any(axis=1))[:max_pushes]
        if idx.size == 0:
            return ref

        # one combined push per sample, so two obstacles at the same sample give one point
        p_push = pos[idx].copy()
        p_push[:, :2] += push_vectors(
            pos[idx], tangent[idx], obstacles, dist[idx], mask[idx], self._clearance
        )
        return ref.update_extras(s[idx], p_push, tangent[idx], ref(s[idx], nu=2))
