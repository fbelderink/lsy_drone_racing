"""Centerline through the gate centers, the reference path of MPCC++ (Krinner, Romero et al., 2024).

The centerline is a Hermite spline that passes every gate center along the gate's normal. It is
the axis of the tunnel of `mpcc/corridor.py`, and the controller is free to deviate from it within
the tunnel. The paper's tracks have no obstacles; here the centerline bends around the poles, so
the tunnel does not lead the drone straight into one, e.g. when a pole stands in front of a gate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.planner.pmm import TURN_EXIT
from lsy_drone_racing.control.planner.reference import Reference
from lsy_drone_racing.control.utils.collision import GATE_OUTER_HALF_WIDTH, gate_contact
from lsy_drone_racing.control.utils.utils import drone_collision_radius, obstacle_radius

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

OVERSHOOT = 1.0  # Length in meters of the straight extension past the last gate
# Horizontal clearance in meters between the drone's and a pole's collision spheres that the
# centerline keeps
POLE_MARGIN = 0.05
# Distance in meters beyond the drone's radius from a gate's frame within which the centerline
# counts as close to the frame. A detour around a pole goes to the side where less of the centerline
# is close to a frame, e.g. not between the pole and a gate's plane.
FRAME_MARGIN = 0.1
# Extra distance in meters by which a detour knot is pushed out, so the spline through it keeps the
# clearance also next to the knot
DETOUR_EXTRA = 0.03
# Clearance in meters between a turnaround knot and the drone touching the gate's outer frame edge
TURN_CLEARANCE = 0.15
MAX_DETOURS = 8  # Maximum number of detour knots added in one build
CHECK_SPACING = 0.01  # Arc-length spacing in meters at which the centerline is checked for poles


def turnaround_knot(
    center: NDArray[np.floating],
    direction: NDArray[np.floating],
    next_center: NDArray[np.floating],
    side_offset: float,
    poles_pos: NDArray[np.floating],
    pole_clearance: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Knot behind a gate from which the centerline turns back around the gate's frame.

    Like `pmm.turnaround_waypoint`, it lies TURN_EXIT behind the gate's plane, moved by side_offset
    to the left, to the right or up, whichever points most towards the next gate, never down to the
    gate's stand. Sides whose knot would be within pole_clearance of a pole are skipped, unless all
    are, since a pole often stands right behind a gate.

    Args:
        center: Gate center with shape (3,).
        direction: Crossing direction of the gate, a unit vector with shape (3,).
        next_center: Center of the next gate with shape (3,).
        side_offset: Distance in meters of the knot from the gate's axis, beyond the frame.
        poles_pos: Pole positions with shape (n_poles, 3).
        pole_clearance: Required horizontal distance in meters from a pole's axis.

    Returns:
        - The knot with shape (3,).
        - The unit side direction with shape (3,), the centerline's direction at the knot.
    """
    horizontal = np.cross(direction, [0.0, 0.0, 1.0])
    horizontal /= la.norm(horizontal)
    sides = np.array([[0.0, 0.0, 1.0], horizontal, -horizontal])
    knots = center + TURN_EXIT * direction + side_offset * sides
    pole_distance = la.norm(knots[:, None, :2] - poles_pos[None, :, :2], axis=-1)
    clear = np.all(pole_distance >= pole_clearance, axis=1)
    preference = sides @ (next_center - center)
    if np.any(clear):
        preference = np.where(clear, preference, -np.inf)
    best = int(np.argmax(preference))
    return knots[best], sides[best]


def centerline_knots(
    start: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_direction: NDArray[np.floating],
    overshoot: float,
    turn_side_offset: float,
    poles_pos: NDArray[np.floating],
    pole_clearance: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
    """Knots and unit tangents of the centerline through the gates.

    The knots are the start, the gate centers and a point `overshoot` past the last gate, so the
    planning horizon does not run off the path's end right after the last gate. When the next gate
    lies behind a gate's plane, a turnaround knot behind the gate is added, see `turnaround_knot`.
    Without it the spline turns around right behind the gate and comes
    back through the gate's own opening or frame.

    Args:
        start: Start position with shape (3,).
        gates_pos: Gate centers in the order they are passed, with shape (n, 3).
        gates_direction: Unit directions in which the gates are passed, with shape (n, 3).
        overshoot: Distance in meters of the last knot past the last gate.
        turn_side_offset: Distance in meters of a turnaround knot from the gate's axis, to the side
            or up, beyond the frame.
        poles_pos: Pole positions with shape (n_poles, 3).
        pole_clearance: Required horizontal distance in meters of a turnaround knot from a pole.

    Returns:
        - The knots with shape (m, 3).
        - The unit tangents with shape (m, 3): towards the first gate at the start, then the gates'
          directions, and at a turnaround knot the side it moves to.
        - Whether each knot is a gate center, with shape (m,).
    """
    to_first_gate = gates_pos[0] - start
    knots, tangents, is_gate = [start], [to_first_gate / la.norm(to_first_gate)], [False]
    for i, (center, direction) in enumerate(zip(gates_pos, gates_direction)):
        knots.append(center)
        tangents.append(direction)
        is_gate.append(True)
        is_last = i + 1 == len(gates_pos)
        if not is_last and np.dot(gates_pos[i + 1] - center, direction) < 0:
            turn, side = turnaround_knot(
                center, direction, gates_pos[i + 1], turn_side_offset, poles_pos, pole_clearance
            )
            knots.append(turn)
            tangents.append(side)
            is_gate.append(False)
    knots.append(gates_pos[-1] + overshoot * gates_direction[-1])
    tangents.append(gates_direction[-1])
    is_gate.append(False)
    return np.array(knots), np.array(tangents), np.array(is_gate)


def closest_pole_conflict(
    reference: Reference, poles_pos: NDArray[np.floating], clearance: float
) -> tuple[float, int, NDArray[np.floating]] | None:
    """Find the deepest point where the centerline comes closer to a pole than the clearance.

    Poles are vertical, so only the horizontal distance counts.

    Args:
        reference: The centerline.
        poles_pos: Pole positions with shape (n_poles, 3).
        clearance: Required horizontal distance in meters from a pole's axis.

    Returns:
        None if the centerline keeps the clearance everywhere. Otherwise the arc length in meters of
        the deepest point, the index of the pole, and two candidate detour knots with shape (2, 3):
        that point pushed
        horizontally and perpendicular to the path to the clearance plus DETOUR_EXTRA from the pole,
        first on the side of the pole the path is already on, then on the other side.
    """
    s = np.arange(0.0, reference.length, CHECK_SPACING)
    points = reference.path(s)
    offsets = points[:, None, :2] - poles_pos[None, :, :2]  # (n_points, n_poles, 2)
    distance = la.norm(offsets, axis=-1)
    i, pole = np.unravel_index(np.argmin(distance - clearance), distance.shape)
    if distance[i, pole] >= clearance:
        return None

    # Push sideways, perpendicular to the path. Pushing along the offset to the pole instead fails
    # when the path runs through the pole's axis, where that offset's direction is arbitrary.
    tangent = reference.path(s[i], 1)[:2]
    tangent = tangent / la.norm(tangent)
    left = np.array([-tangent[1], tangent[0]])
    side = 1.0 if np.dot(offsets[i, pole], left) >= 0 else -1.0
    along = np.dot(offsets[i, pole], tangent)
    knots = np.tile(points[i], (2, 1))
    for knot, knot_side in zip(knots, (side, -side)):
        knot[:2] = (
            poles_pos[pole, :2] + along * tangent + knot_side * (clearance + DETOUR_EXTRA) * left
        )
    return float(s[i]), int(pole), knots


def gate_frame_proximity(reference: Reference, gates: NDArray[np.floating], radius: float) -> float:
    """Length of the centerline that comes closer than radius to a gate's frame or stand.

    Args:
        reference: The centerline.
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        radius: Distance in meters from the frame that counts as close.

    Returns:
        The length in meters.
    """
    points = reference.path(np.arange(0.0, reference.length, CHECK_SPACING))
    return CHECK_SPACING * float(np.sum(gate_contact(points, gates, radius)))


class CenterlinePlanner:
    """Plans the centerline from the start position through the gates in their sequence.

    When a gate's or a pole's true pose is revealed, the whole centerline is rebuilt from the same
    start, so the tunnel always describes the track and not the drone's current state.
    """

    # The centerline always starts at the drone's initial position, so the progress is projected
    # onto a new centerline instead of restarting at zero
    starts_at_drone = False

    def __init__(self, obs: dict[str, NDArray[np.floating]], config: dict):
        """Plan the centerline through the gates' and poles' poses in the initial observation.

        Args:
            obs: The initial observation of the environment's state.
            config: The race configuration.
        """
        self._start = np.array(obs["pos"], dtype=float)
        self._gate_sequence = np.asarray(obs["gate_sequence"])
        self._gate_sequence_direction = np.asarray(obs["gate_sequence_direction"], dtype=float)
        self._gates_visited = np.array(obs["gates_visited"])
        self._obstacles_visited = np.array(obs["obstacles_visited"])
        self._pole_clearance = drone_collision_radius(config.sim.drone) + obstacle_radius()
        self._pole_clearance += POLE_MARGIN
        self._frame_radius = drone_collision_radius(config.sim.drone) + FRAME_MARGIN
        self._turn_side_offset = (
            GATE_OUTER_HALF_WIDTH + drone_collision_radius(config.sim.drone) + TURN_CLEARANCE
        )
        # Horizontal unit direction from each pole to its detour, kept across rebuilds
        self._detour_sides: dict[int, NDArray[np.floating]] = {}
        self._reference = None
        self.gate_progress = None  # Arc length of every gate crossing in meters, shape (n,)
        self._build(obs)

    def plan(self, obs: dict[str, NDArray[np.floating]]) -> Reference:
        """Return the centerline, rebuilt if the observation revealed a gate's or pole's pose.

        Args:
            obs: The current observation of the environment.

        Returns:
            The current centerline.
        """
        new_gates = np.any(obs["gates_visited"] & ~self._gates_visited)
        new_obstacles = np.any(obs["obstacles_visited"] & ~self._obstacles_visited)
        if new_gates or new_obstacles:
            self._gates_visited = np.array(obs["gates_visited"])
            self._obstacles_visited = np.array(obs["obstacles_visited"])
            self._build(obs)
        return self._reference

    def _build(self, obs: dict[str, NDArray[np.floating]]):
        """Build the centerline through the gate poses, around the poles of the observation."""
        gates_pos = np.asarray(obs["gates_pos"])[self._gate_sequence]
        normals = R.from_quat(np.asarray(obs["gates_quat"])[self._gate_sequence]).apply([1, 0, 0])
        directions = normals * self._gate_sequence_direction[:, None]
        poles_pos = np.asarray(obs["obstacles_pos"])
        knots, tangents, is_gate = centerline_knots(
            self._start,
            gates_pos,
            directions,
            OVERSHOOT,
            self._turn_side_offset,
            poles_pos,
            self._pole_clearance + DETOUR_EXTRA,
        )

        all_normals = R.from_quat(np.asarray(obs["gates_quat"])).apply([1, 0, 0])
        gates = np.hstack((np.asarray(obs["gates_pos"]), all_normals))
        reference = Reference(knots, tangents, np.zeros_like(tangents))
        conflict = closest_pole_conflict(reference, poles_pos, self._pole_clearance)
        n_detours = 0
        while conflict is not None and n_detours < MAX_DETOURS:
            s, pole, candidates = conflict
            # Insert the detour knot between the knots around it, keeping the local direction
            index = int(np.searchsorted(reference.sparse_s, s))
            tangent = reference.path(s, 1)
            tangents = np.insert(tangents, index, tangent / la.norm(tangent), axis=0)
            is_gate = np.insert(is_gate, index, False)
            if pole in self._detour_sides:
                # Keep the side chosen in an earlier build, so the centerline does not switch to the
                # pole's other side while the drone is flying past it
                away = candidates[:, :2] - poles_pos[pole, :2]
                candidates = candidates[[int(np.argmax(away @ self._detour_sides[pole]))]]
            options = []
            for knot in candidates:
                option_knots = np.insert(knots, index, knot, axis=0)
                option = Reference(option_knots, tangents, np.zeros_like(tangents))
                proximity = gate_frame_proximity(option, gates, self._frame_radius)
                options.append((proximity, option.length, option_knots, option, knot))
            # Least of the centerline near a gate frame, then the shortest
            _, _, knots, reference, knot = min(options, key=lambda option: option[:2])
            side = knot[:2] - poles_pos[pole, :2]
            self._detour_sides.setdefault(pole, side / la.norm(side))
            conflict = closest_pole_conflict(reference, poles_pos, self._pole_clearance)
            n_detours += 1
        if conflict is not None:
            logger.warning("Centerline still passes a pole after %d detours", MAX_DETOURS)

        if self._reference is None:
            self._reference = reference
        else:
            # Rebuilt in place, which increments the reference's version
            self._reference._update(knots, tangents, np.zeros_like(tangents))
        self.gate_progress = self._reference.sparse_s[is_gate]
