"""Find where a planned path collides with the track and move a new waypoint out of the way.

A planner that ignores obstacles can use these functions in a loop: plan, find the first
collision, insert a waypoint pushed out of the obstacle or gate there, and plan again.

Obstacles are not avoided inside the corridor right in front of and behind a gate's opening: the
path has to cross the gate straight there, and steering it around a nearby pole would make it
cross the gate's plane off-center, into the frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la

from lsy_drone_racing.control.utils.collision import (
    GATE_INNER_HALF_WIDTH,
    GATE_OUTER_HALF_WIDTH,
    GATE_STAND_HALF_WIDTH,
    gate_contact,
    obstacle_clearance,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Clearance in meters the path must keep, and the clearance of a pushed-out waypoint (larger, so
# the repaired path does not end up at the limit). Poles get more clearance than gate frames: the
# drone may cut corners around them at speed, while the gate opening leaves room for only a few
# centimeters (0.2 m half-width minus the drone's radius).
OBSTACLE_CHECK_MARGIN = 0.12
OBSTACLE_PUSH_MARGIN = 0.2
GATE_CHECK_MARGIN = 0.03
GATE_PUSH_MARGIN = 0.06
START_SKIP = 0.15  # Length in meters of the path's start that is not checked, near the drone
# Length in meters of the path that is checked. Parts further ahead are repaired when the planner
# re-plans on the next revealed gate or obstacle, which keeps the number of repairs per plan small.
CHECK_DISTANCE = 4.0
# Half-length in meters along the gate's normal of the corridor in which obstacles are not avoided,
# covering the planner's before/after waypoints of a gate
GATE_CORRIDOR_HALF_LENGTH = 0.35
# Smallest distance in meters along the gate's normal to which a gate's before/after waypoint is
# slid towards the gate's center to clear an obstacle
MIN_GATE_OFFSET = 0.05
# Distance in meters of the two waypoints that straighten a path crossing a gate it is not meant to
# pass, before and after the gate's plane
CROSSING_OFFSET = 0.2


def gate_corridors(points: NDArray[np.floating], gates: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Check which points lie in the straight corridor through each gate's opening.

    Args:
        points: Points with shape (n, 3).
        gates: Position and normal vector of every gate with shape (n_gates, 6).

    Returns:
        Flags with shape (n, n_gates).
    """
    offset = points[:, None, :] - gates[None, :, :3]
    normal = gates[:, 3:]
    horizontal_axis = np.cross(normal, [0.0, 0.0, 1.0])
    along_normal = np.abs(np.sum(offset * normal, axis=-1))
    horizontal = np.abs(np.sum(offset * horizontal_axis, axis=-1))
    in_plane = np.maximum(horizontal, np.abs(offset[..., 2]))
    return (along_normal < GATE_CORRIDOR_HALF_LENGTH) & (in_plane < GATE_INNER_HALF_WIDTH)


def in_gate_corridor(
    points: NDArray[np.floating], gates: NDArray[np.floating]
) -> NDArray[np.bool_]:
    """Check which points lie in the straight corridor through any gate's opening.

    Args:
        points: Points with shape (n, 3).
        gates: Position and normal vector of every gate with shape (n_gates, 6).

    Returns:
        Flags with shape (n,).
    """
    return np.any(gate_corridors(points, gates), axis=-1)


def slide_toward_gate(
    point: NDArray[np.floating],
    gate: NDArray[np.floating],
    obstacles_xy: NDArray[np.floating],
    min_distance: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Slide a point in a gate's corridor along the gate's normal towards its center until clear.

    Moving along the normal keeps the gate crossing straight, unlike pushing the point sideways.

    Args:
        point: The point with shape (3,).
        gate: Position and normal vector of the gate with shape (6,).
        obstacles_xy: Horizontal obstacle positions with shape (n_obstacles, 2).
        min_distance: Required horizontal distance in meters from each obstacle's axis with shape
            (n_obstacles,).

    Returns:
        The first clear point, or the one MIN_GATE_OFFSET from the gate's plane, with shape (3,).
    """
    center, normal = gate[:3], gate[3:]
    along = (point - center) @ normal
    across = point - center - along * normal
    for distance in np.linspace(abs(along), MIN_GATE_OFFSET, 20):
        candidate = center + across + np.sign(along) * distance * normal
        if np.all(la.norm(obstacles_xy - candidate[:2], axis=1) >= min_distance):
            break
    return candidate


def first_collision(
    points: NDArray[np.floating],
    gates: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    obstacles_inflation: NDArray[np.floating],
    drone_radius: float,
    obstacle_radius: float,
) -> int | None:
    """Middle point of the first stretch of the path that comes too close to a gate or obstacle.

    The middle of the stretch is pushed out rather than its first point, so that one waypoint
    clears the whole stretch instead of the path creeping around the obstacle. The first
    START_SKIP meters are not checked: the drone is already there, and a waypoint inserted next to
    it cannot move it. Neither is the path beyond CHECK_DISTANCE.

    Args:
        points: Path points with shape (n, 3).
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3).
        obstacles_inflation: Extra obstacle radius in meters with shape (n_obstacles,).
        drone_radius: Collision radius of the drone in meters.
        obstacle_radius: Collision radius of the obstacle poles in meters.

    Returns:
        The index into points, or None if the path is free.
    """
    radius = drone_radius + obstacle_radius + OBSTACLE_CHECK_MARGIN
    clearance = obstacle_clearance(points, obstacles_pos, obstacles_inflation, radius)
    near_obstacle = (clearance < 0) & ~in_gate_corridor(points, gates)
    collides = near_obstacle | gate_contact(points, gates, drone_radius + GATE_CHECK_MARGIN)
    distance = np.append(0.0, np.cumsum(la.norm(np.diff(points, axis=0), axis=1)))
    collides &= (distance > START_SKIP) & (distance < CHECK_DISTANCE)
    hits = np.flatnonzero(collides)
    if hits.size == 0:
        return None
    # End of the first stretch of consecutive colliding points
    gaps = np.flatnonzero(np.diff(hits) > 1)
    last = hits[gaps[0]] if gaps.size else hits[-1]
    return int((hits[0] + last) // 2)


def push_out_of_obstacle(
    point: NDArray[np.floating], obstacle_xy: NDArray[np.floating], distance: float
) -> NDArray[np.floating]:
    """Move a point horizontally away from a pole to the given distance from its axis.

    Args:
        point: The point with shape (3,).
        obstacle_xy: Horizontal position of the pole with shape (2,).
        distance: Distance in meters from the pole's axis.

    Returns:
        The moved point with shape (3,).
    """
    away = point[:2] - obstacle_xy
    if la.norm(away) < 1e-9:  # On the axis: any direction is as good
        away = np.array([1.0, 0.0])
    moved = point.copy()
    moved[:2] = obstacle_xy + distance * away / la.norm(away)
    return moved


def push_out_of_gate(
    point: NDArray[np.floating], gate: NDArray[np.floating], drone_radius: float
) -> NDArray[np.floating]:
    """Move a point in a gate's frame or stand to the nearest free space beside it.

    A point on the frame moves into the opening or beyond the frame's outer edge, whichever is
    closer, but never below the frame, where the stand is. A point at the stand moves sideways.

    Args:
        point: The point with shape (3,).
        gate: Position and normal vector of the gate with shape (6,).
        drone_radius: Collision radius of the drone in meters.

    Returns:
        The moved point with shape (3,).
    """
    center, normal = gate[:3], gate[3:]
    horizontal_axis = np.cross(normal, [0.0, 0.0, 1.0])
    horizontal_axis /= la.norm(horizontal_axis)
    offset = point - center
    along_normal = offset @ normal
    horizontal = offset @ horizontal_axis
    vertical = offset[2]
    inside = GATE_INNER_HALF_WIDTH - drone_radius - GATE_PUSH_MARGIN
    outside = GATE_OUTER_HALF_WIDTH + drone_radius + GATE_PUSH_MARGIN
    side = 1.0 if horizontal >= 0 else -1.0

    in_plane = max(abs(horizontal), abs(vertical))
    if vertical < -GATE_OUTER_HALF_WIDTH:  # Stand
        horizontal = side * (GATE_STAND_HALF_WIDTH + drone_radius + GATE_PUSH_MARGIN)
    elif in_plane - inside < outside - in_plane or vertical < -abs(horizontal):
        # Into the opening, keeping the direction from the center
        horizontal, vertical = np.array([horizontal, vertical]) * inside / max(in_plane, 1e-9)
    elif abs(horizontal) >= abs(vertical):
        horizontal = side * outside
    else:
        vertical = outside
    return (
        center
        + along_normal * normal
        + horizontal * horizontal_axis
        + vertical * np.array([0.0, 0.0, 1.0])
    )


def insertion_index(points: NDArray[np.floating], waypoints: NDArray[np.floating], hit: int) -> int:
    """Index in the waypoint list at which a waypoint for the path point `hit` belongs.

    The path passes through the waypoints in order, so each waypoint is matched to its closest
    path point, searching forward from the previous waypoint's match.

    Args:
        points: Path points with shape (n, 3), starting at the drone.
        waypoints: Waypoints after the start with shape (n_waypoints, 3).
        hit: Index of the colliding path point.

    Returns:
        The number of waypoints the path passes before reaching points[hit].
    """
    matched = []
    start = 0
    for waypoint in waypoints:
        start += int(np.argmin(la.norm(points[start:] - waypoint, axis=1)))
        matched.append(start)
    return int(np.searchsorted(matched, hit))


def insert_repair_waypoints(
    waypoints: NDArray[np.floating], index: int, new_waypoints: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Insert the waypoints of `repair_path_hit` into the waypoint list.

    A straight gate crossing (two new waypoints) replaces the waypoints that lie on it, within the
    gate's opening: the gate's center or an earlier crossing of the same gate. Inserted next to
    them instead, repeated repairs of one gate stack crossings and the path runs back and forth.

    Args:
        waypoints: Waypoints with shape (n_waypoints, 3).
        index: Position in `waypoints` at which the new waypoints belong, see `insertion_index`.
        new_waypoints: Waypoints in path order with shape (n_new, 3).

    Returns:
        The new waypoint list with shape (n_waypoints', 3).
    """
    if len(new_waypoints) == 2:
        start, end = new_waypoints
        segment = end - start
        t = (waypoints - start) @ segment / (segment @ segment)
        closest = start + np.clip(t, 0.0, 1.0)[:, None] * segment
        on_crossing = (t >= 0.0) & (t <= 1.0)
        on_crossing &= la.norm(waypoints - closest, axis=1) < GATE_INNER_HALF_WIDTH
        index = int(np.sum(~on_crossing[:index]))
        waypoints = waypoints[~on_crossing]
    return np.insert(waypoints, [index] * len(new_waypoints), new_waypoints, axis=0)


def repair_waypoint(
    point: NDArray[np.floating],
    gates: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    obstacles_inflation: NDArray[np.floating],
    drone_radius: float,
    obstacle_radius: float,
) -> NDArray[np.floating]:
    """New waypoint for a colliding path point, pushed out of what it collides with.

    Args:
        point: The colliding path point with shape (3,).
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3).
        obstacles_inflation: Extra obstacle radius in meters with shape (n_obstacles,).
        drone_radius: Collision radius of the drone in meters.
        obstacle_radius: Collision radius of the obstacle poles in meters.

    Returns:
        The waypoint with shape (3,).
    """
    radius = drone_radius + obstacle_radius
    distance = la.norm(obstacles_pos[:, :2] - point[:2], axis=1) - obstacles_inflation
    nearest = int(np.argmin(distance))
    if distance[nearest] < radius + OBSTACLE_CHECK_MARGIN:
        corridor = np.flatnonzero(gate_corridors(point[None], gates)[0])
        if corridor.size:  # A gate's before/after waypoint: keep the crossing straight
            min_distance = radius + obstacles_inflation + OBSTACLE_CHECK_MARGIN
            return slide_toward_gate(point, gates[corridor[0]], obstacles_pos[:, :2], min_distance)
        push = radius + obstacles_inflation[nearest] + OBSTACLE_PUSH_MARGIN
        return push_out_of_obstacle(point, obstacles_pos[nearest, :2], push)
    # gate_contact reduces over gates, so check them one by one for the one in contact
    for gate in gates:
        if bool(gate_contact(point[None], gate[None], drone_radius + GATE_CHECK_MARGIN)[0]):
            return push_out_of_gate(point, gate, drone_radius)
    return point  # Not colliding


def repair_path_hit(
    points: NDArray[np.floating],
    hit: int,
    gates: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    obstacles_inflation: NDArray[np.floating],
    drone_radius: float,
    obstacle_radius: float,
) -> NDArray[np.floating]:
    """Waypoints that move the path out of a collision at points[hit].

    Usually a single waypoint, see `repair_waypoint`. When the path cuts through a gate's frame and
    the nearest free space is the opening, one point there cannot straighten the oblique crossing.
    Two waypoints are returned instead, before and after the gate's plane along its normal, in the
    direction the path crosses it.

    Args:
        points: Path points with shape (n, 3).
        hit: Index of the colliding path point, see `first_collision`.
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3).
        obstacles_inflation: Extra obstacle radius in meters with shape (n_obstacles,).
        drone_radius: Collision radius of the drone in meters.
        obstacle_radius: Collision radius of the obstacle poles in meters.

    Returns:
        The waypoints in path order with shape (n_new, 3).
    """
    args = (gates, obstacles_pos, obstacles_inflation, drone_radius, obstacle_radius)
    point = points[hit]
    moved = repair_waypoint(point, *args)
    in_frame = gate_contact(point[None], gates, drone_radius + GATE_CHECK_MARGIN)[0]
    corridor = np.flatnonzero(gate_corridors(moved[None], gates)[0])
    if not (in_frame and corridor.size):
        return moved[None]
    gate = gates[corridor[0]]
    center, normal = gate[:3], gate[3:]
    direction = points[min(hit + 1, len(points) - 1)] - points[max(hit - 1, 0)]
    crossing = normal if direction @ normal >= 0 else -normal
    in_plane = moved - ((moved - center) @ normal) * normal
    return np.stack((in_plane - CROSSING_OFFSET * crossing, in_plane + CROSSING_OFFSET * crossing))
