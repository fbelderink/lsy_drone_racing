import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray

from lsy_drone_racing.control.planner.reference import ProgressReference
from lsy_drone_racing.control.utils.types import ObservationDict
from lsy_drone_racing.control.utils.utils import drone_collision_radius, obstacle_radius


def check_gate_collision(ref: NDArray, gate_pose: NDArray, clearance: float) -> NDArray:
    """Args:
    ref (N, 4): [s, x, y, z]
    gate_pos (G, 6) [x, y, z, roll, pitch, yaw]
    clearance: Minimum clearance between every point on the reference and the gates geometry.
    """
    pass


def check_obstacle_collision(ref: NDArray, obstacle_pos: NDArray, clearance: float) -> NDArray:
    """Find points on reference which are too close to obstacles.

    Args:
        ref (N, 3): [x, y, z]
        obstacle_pos (O, 3): [x, y, z]
        clearance: Minimum clearance between every point on the reference and the gates geometry.

    Returns:
        distances to obstacles, mask (N, O) where reference is too close to obstacle
    """
    d = ref[:, None, :] - obstacle_pos[None, :, :]  # (N, O, 3)
    dist_xy = la.norm(d[..., :2], axis=-1)  # (N, O)
    above_pole = d[..., 2] > clearance
    dist = np.where(above_pole, np.inf, dist_xy)  # (N, O)

    _, num_obs = dist.shape

    mask = dist <= clearance
    pad = np.full((1, num_obs), np.inf)

    d_in = np.where(mask, dist, np.inf)
    prev = np.vstack([pad, d_in[:-1]])
    nxt = np.vstack([d_in[1:], pad])

    closest_collisions = (
        mask & (d_in <= prev) & (d_in < nxt)
    )  # only keep the local minima of each group

    return dist, closest_collisions


class CollisionAvoidanceStrategy:
    """Strategy for High Level Planner to avoid collisions in path."""

    def __init__(self, cfg: dict, n_iters: int = 3):
        """Initialize Collision Avoidance Strategy."""
        self._margin = 0.04  # [m] safety margin for collisions

        self._d_r = drone_collision_radius(cfg.sim.drone)
        self._o_r = obstacle_radius()

        self.n_iters = n_iters

    def avoid_collisions(self, obs: ObservationDict, ref: ProgressReference) -> ProgressReference:
        for _ in range(self.n_iters):
            ref = self.avoid_obstacle_collision(obs, ref)

        return ref

    def avoid_obstacle_collision(
        self,
        obs: ObservationDict,
        ref: ProgressReference,
        n_samples: int = 1000,
        max_pushes: int = 8,
        eps: float = 0.01,
    ) -> ProgressReference:
        """Push the reference away from obstacles.

        Every iteration takes the local distance minima per obstacle (check_obstacle_collision),
        combines obstacles that hit the same sample into one push, and inserts up to max_pushes
        push points. Shapes are fixed (N, O or max_pushes) and the loop count is fixed, so the
        structure carries over to jax.
        """
        clearance = self._d_r + self._o_r + self._margin
        obstacles = obs["obstacles_pos"]

        uniform_s = np.linspace(0, ref.length, n_samples)
        ref_pos, ref_tangent = ref(uniform_s, nu=0), ref(uniform_s, nu=1)  # (N, 3)
        ds = uniform_s[1] - uniform_s[0]

        # inflate by ds/2: then clear samples guarantee a clear path in between
        dist, collision_mask = check_obstacle_collision(
            ref_pos, obstacles, clearance=clearance + ds / 2
        )  # (N, O), (N, O)
        collision_mask &= obs["obstacles_visited"][None, :]

        # (N, O) -> (N,): one combined push per sample, so two obstacles at the same
        # sample give one point instead of two conflicting ones
        push = push_vectors(ref_pos, ref_tangent, obstacles, dist, collision_mask, clearance + eps)

        # (N,) -> (K,): fixed number of candidates, update_extras ignores the invalid ones,
        # so no early return is needed when nothing collides
        idx, valid = fixed_nonzero(collision_mask.any(axis=1), max_pushes)  # (K,), (K,)
        s_push = uniform_s[idx]
        p_push = ref_pos[idx].copy()
        p_push[:, :2] += push[idx]

        return ref.update_extras(s_push, p_push, ref_tangent[idx], ref(s_push, nu=2), valid=valid)


def push_vectors(
    pos: NDArray, tangent: NDArray, obstacles: NDArray, dist: NDArray, mask: NDArray, target: float
) -> NDArray:
    """Horizontal push per sample that moves it to `target` distance from every masked obstacle.

    The pushes of all obstacles masked at the same sample are summed. Obstacles on the same side
    add up (conservative), obstacles on opposite sides cancel, which leaves the point in between.

    Args:
        pos: (N, 3) path positions.
        tangent: (N, 3) unit tangents of the path.
        obstacles: (O, 3) obstacle positions.
        dist: (N, O) horizontal distance to the pole axes (inf where irrelevant).
        mask: (N, O) samples to push away from each obstacle.
        target: Distance from the pole axis the pushed point should have.

    Returns:
        (N, 2) push in xy, zero where nothing is masked.
    """
    away = pos[:, None, :2] - obstacles[None, :, :2]  # (N, O, 2) pole axis → path
    # if the path passes through the axis exactly, push along the horizontal normal of the path
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)[:, None, :]  # (N, 1, 2)
    normal = normal / np.maximum(la.norm(normal, axis=-1, keepdims=True), 1e-9)
    norm = la.norm(away, axis=-1, keepdims=True)
    away = np.where(norm > 1e-6, away / np.maximum(norm, 1e-9), normal)  # (N, O, 2)

    deficit = np.where(mask, target - dist, 0.0)  # (N, O) how far to push, 0 if not masked
    return np.sum(deficit[..., None] * away, axis=1)  # (N, 2)


def fixed_nonzero(mask: NDArray, size: int) -> tuple[NDArray, NDArray]:
    """Indices of the True entries, padded to a fixed size (like jnp.nonzero(..., size=size)).

    Args:
        mask: (N,) bool.
        size: Number of returned indices. Entries beyond the number of True values are padded
            with index 0 and marked invalid; True values beyond size are dropped.

    Returns:
        idx (size,) indices, valid (size,) bool.
    """
    idx = np.argsort(~mask, kind="stable")[:size]  # True entries first, in order
    valid = np.arange(size) < mask.sum()
    return np.where(valid, idx, 0), valid
