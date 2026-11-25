"""Collision avoidance for the high level planner's reference path.

The reference is sampled densely and checked against the obstacles and gates seen so far. Each
check describes its collisions the same way: per sample and object, the push that clears it. The
deepest sample of each colliding stretch is pushed out and inserted as an extra point into the
reference. This is repeated a few times, since a pushed path can collide again.
"""

from typing import NamedTuple

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.planner.reference import ProgressReference
from lsy_drone_racing.control.utils.types import ObservationDict
from lsy_drone_racing.control.utils.utils import (
    GateGeometry,
    drone_collision_radius,
    gate_geometry,
    obstacle_radius,
)


class Collisions(NamedTuple):
    """Collisions of N consecutive path samples with K objects."""

    push: NDArray  # (N, K, 3) displacement that clears the object, zero where clear
    depth: NDArray  # (N, K) how deep a sample is in the collision, -inf where clear
    direction: NDArray  # (N, K, 3) direction a pushed sample must cross in, zero where free


def obstacle_collisions(
    pos: NDArray,
    tangent: NDArray,
    obstacles: NDArray,
    radius: NDArray,
    clearance: float,
    margin: float = 0.0,
) -> Collisions:
    """Collisions with the obstacle poles, pushed horizontally away from the pole axes.

    Args:
        pos: (N, 3) consecutive points of the path.
        tangent: (N, 3) unit tangents of the path.
        obstacles: (O, 3) obstacle positions, the tops of the poles.
        radius: (O, 1) obstacle radii.
        clearance: Horizontal distance to a pole axis below which a sample collides.
        margin: Distance by which colliding samples are pushed beyond the clearance.

    Returns:
        The collisions, the deepest sample being the closest one to the pole.
    """
    d = pos[:, None, :] - obstacles[None, :, :]  # (N, O, 3)
    dist = np.where(d[..., 2] > clearance, np.inf, la.norm(d[..., :2], axis=-1))  # inf above
    axis_clearance = radius + clearance
    hits = dist <= axis_clearance

    away = d[..., :2] / np.maximum(dist, 1e-9)[..., None]  # (N, O, 2) pole axis -> path
    # if the path passes through the axis exactly, push along the horizontal normal of the path
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    normal /= np.maximum(la.norm(normal, axis=-1, keepdims=True), 1e-9)
    away = np.where(dist[..., None] > 1e-6, away, normal[:, None, :])

    push = np.zeros_like(d)
    push[..., :2] = np.where(hits, axis_clearance + margin - dist, 0.0)[..., None] * away
    return Collisions(push, np.where(hits, -dist, -np.inf), np.zeros_like(d))


def gate_collisions(
    pos: NDArray,
    gates_pos: NDArray,
    gates_quat: NDArray,
    gate: GateGeometry,
    clearance: float,
    margin: float = 0.0,
) -> Collisions:
    """Collisions with the gates' frames (see frame_collisions), in world coordinates.

    Args:
        pos: (N, 3) consecutive points of the path.
        gates_pos: (G, 3) gate centers.
        gates_quat: (G, 4) gate orientations as (x, y, z, w) quaternions.
        gate: Frame geometry of the gates.
        clearance: Minimum distance to the frame (drone radius + margin).
        margin: Distance by which colliding samples are pushed beyond the clearance.

    Returns:
        The collisions, with the gate's normal as direction where the path crosses its plane.
    """
    pushes, depths, directions = [], [], []
    for center, quat in zip(gates_pos, gates_quat):
        rot = R.from_quat(quat)  # gate frame -> world
        push, depth, crossing = frame_collisions(
            rot.inv().apply(pos - center), gate, clearance, margin
        )
        pushes.append(rot.apply(push))
        depths.append(depth)
        directions.append(np.where(crossing[:, None], rot.apply([1.0, 0.0, 0.0]), 0.0))

    return Collisions(
        np.stack(pushes, axis=1), np.stack(depths, axis=1), np.stack(directions, axis=1)
    )


def frame_collisions(
    local: NDArray, gate: GateGeometry, clearance: float, margin: float = 0.0
) -> tuple[NDArray, NDArray, NDArray]:
    """Collisions with a gate's frame, given in the gate's frame (x: normal, y, z: in-plane).

    The frame is a square ring around the opening, so points passing through the opening are
    free. A colliding point gets the shortest of three pushes: in the plane into the opening, in
    the plane past the outer edge, or along the normal off the plane. The last one is only
    allowed where the path stays on one side of the plane: for a path crossing it, moving one
    point off the plane leaves the crossing next to it.

    Args:
        local: (N, 3) consecutive points of the path in the gate's frame.
        gate: Frame geometry of the gate.
        clearance: Minimum distance to the frame.
        margin: Distance by which colliding points are pushed beyond the clearance, so that the
            path next to a pushed point clears the frame as well.

    Returns:
        push (N, 3) in the gate's frame, depth (N,) of each point in the frame (-inf where
        clear), and crossing (N,) where the colliding stretch crosses the plane.
    """
    x, yz = local[:, 0], local[:, 1:]
    inner, outer = gate.inner - clearance, gate.outer + clearance
    thickness = gate.half_depth + clearance
    ring = np.abs(yz).max(axis=1)  # the frame is square, so its rings are max-norm circles
    hits = (np.abs(x) < thickness) & (inner < ring) & (ring < outer)

    into_opening = np.clip(yz, -(inner - margin), inner - margin) - yz
    dominant = np.eye(2)[np.abs(yz).argmax(axis=1)]  # moving the larger coordinate leaves it
    past_edge = dominant * (np.sign(yz) * (outer + margin) - yz)
    off_plane = np.where(x >= 0, 1.0, -1.0) * (thickness + margin) - x

    zero = np.zeros_like(x)
    candidates = np.stack(  # (N, 3 candidates, 3)
        [
            np.column_stack([zero, into_opening]),
            np.column_stack([zero, past_edge]),
            np.column_stack([off_plane, zero, zero]),
        ],
        axis=1,
    )
    length = la.norm(candidates, axis=-1)  # (N, 3)
    crossing = crosses_plane(x, hits)
    length[crossing, 2] = np.inf
    best = candidates[np.arange(len(x)), length.argmin(axis=1)]
    push = np.where(hits[:, None], best, 0.0)

    # a crossing stretch is pushed where it crosses the plane, any other where it is deepest
    depth = np.where(crossing, thickness - np.abs(x), la.norm(push, axis=1))
    depth = np.where(hits, depth, -np.inf)
    return push, depth, crossing


def crosses_plane(x: NDArray, hits: NDArray) -> NDArray:
    """Mask (N,) of the hits whose stretch of consecutive hits has points on both sides of x = 0."""
    stretch = np.cumsum(hits & ~np.append(False, hits[:-1]))  # id of the stretch, from 1
    lowest, highest = np.zeros(stretch[-1] + 1), np.zeros(stretch[-1] + 1)
    np.minimum.at(lowest, stretch[hits], x[hits])
    np.maximum.at(highest, stretch[hits], x[hits])
    return hits & (lowest < 0)[stretch] & (highest > 0)[stretch]


def deepest_points(depth: NDArray) -> NDArray:
    """Mask (N, K) of the deepest sample of each colliding stretch along N, per object K."""
    pad = np.full((1, depth.shape[1]), -np.inf)
    prev = np.concatenate([pad, depth[:-1]])
    nxt = np.concatenate([depth[1:], pad])
    return (depth > -np.inf) & (depth >= prev) & (depth > nxt)


class CollisionAvoidanceStrategy:
    """Strategy for High Level Planner to avoid collisions in path."""

    def __init__(self, cfg: dict, n_iters: int = 3, n_samples: int = 1000):
        """Initialize Collision Avoidance Strategy."""
        margin = 0.01  # [m] safety margin for collisions
        drone_radius = drone_collision_radius(cfg.sim.drone)
        self._clearance = drone_radius + margin
        self._gate = gate_geometry()
        self._obstacle_radius = obstacle_radius()
        self.n_iters = n_iters
        self.n_samples = n_samples

    def avoid_collisions(self, obs: ObservationDict, ref: ProgressReference) -> ProgressReference:
        """Push the reference away from obstacles until it is clear or n_iters is reached."""
        for _ in range(self.n_iters):
            new_ref = self.avoid_obstacle_collision(obs, ref)
            new_ref = self.avoid_gate_collision(obs, new_ref)
            if new_ref is ref:  # nothing collided
                break
            ref = new_ref

        return ref

    def avoid_obstacle_collision(
        self,
        obs: ObservationDict,
        ref: ProgressReference,
        max_pushes: int = 8,
        push_margin: float = 0.01,
    ) -> ProgressReference:
        """Push the reference away from the revealed obstacles."""
        visited = obs["obstacles_visited"]
        if not visited.any():
            return ref

        s, pos, tangent, ds = self._sample(ref)
        # inflate by ds/2: then clear samples guarantee a clear path in between
        collisions = obstacle_collisions(
            pos,
            tangent,
            obs["obstacles_pos"][visited],
            self._obstacle_radius,
            self._clearance + ds / 2,
            push_margin,
        )
        return self._push_out(ref, s, pos, tangent, collisions, max_pushes)

    def avoid_gate_collision(
        self,
        obs: ObservationDict,
        ref: ProgressReference,
        max_pushes: int = 8,
        push_margin: float = 0.01,
    ) -> ProgressReference:
        """Push the reference out of the frames of the revealed gates."""
        visited = obs["gates_visited"]
        if not visited.any():
            return ref

        s, pos, tangent, ds = self._sample(ref)
        # inflate by ds/2: then clear samples guarantee a clear path in between
        collisions = gate_collisions(
            pos,
            obs["gates_pos"][visited],
            obs["gates_quat"][visited],
            self._gate,
            self._clearance + ds / 2,
            push_margin,
        )
        return self._push_out(ref, s, pos, tangent, collisions, max_pushes)

    def _sample(self, ref: ProgressReference) -> tuple[NDArray, NDArray, NDArray, float]:
        """Sample the reference uniformly in progress: s, positions, tangents and spacing ds."""
        s = np.linspace(0, ref.length, self.n_samples)
        return s, ref(s, nu=0), ref(s, nu=1), s[1] - s[0]

    def _push_out(
        self,
        ref: ProgressReference,
        s: NDArray,
        pos: NDArray,
        tangent: NDArray,
        collisions: Collisions,
        max_pushes: int,
    ) -> ProgressReference:
        """Insert the deepest sample of each colliding stretch, pushed out, as extra points.

        The pushes of all objects at the same sample are summed: on the same side they add up
        (conservative), on opposite sides they cancel, which leaves the point in between.
        Returns ref unchanged if nothing collides.
        """
        deepest = deepest_points(collisions.depth)  # (N, K)
        idx = np.flatnonzero(deepest.any(axis=1))[:max_pushes]
        if idx.size == 0:
            return ref

        selected = deepest[idx, :, None]  # (P, K, 1)
        p = pos[idx] + np.sum(np.where(selected, collisions.push[idx], 0.0), axis=1)
        v, a = tangent[idx], ref(s[idx], nu=2)

        # a path crossing a gate's plane is turned to cross it straight, like through a gate:
        # crossing obliquely, it would stay close to the frame along a long stretch of the plane
        direction = np.sum(np.where(selected, collisions.direction[idx], 0.0), axis=1)
        straight = la.norm(direction, axis=1) > 0
        side = np.sign(np.sum(v * direction, axis=1, keepdims=True))
        v = np.where(straight[:, None], side * direction, v)
        a = np.where(straight[:, None], 0.0, a)

        return ref.update_extras(s[idx], p, v, a)
