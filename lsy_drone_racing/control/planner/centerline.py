"""Centerline through the gate centers, the reference path of MPCC++ (Krinner, Romero et al., 2024).

The centerline is a Hermite spline that passes every gate center along the gate's normal. It does
not avoid obstacles or the gate frames: it is the axis of the tunnel of `mpcc/corridor.py`, and the
controller is free to deviate from it within the tunnel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.planner.reference import Reference

if TYPE_CHECKING:
    from numpy.typing import NDArray

OVERSHOOT = 2.0  # Length in meters of the straight extension past the last gate


def centerline_knots(
    start: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_direction: NDArray[np.floating],
    overshoot: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Knots and unit tangents of the centerline.

    The knots are the start, the gate centers and a point `overshoot` past the last gate, so the
    planning horizon does not run off the path's end right after the last gate.

    Args:
        start: Start position with shape (3,).
        gates_pos: Gate centers in the order they are passed, with shape (n, 3).
        gates_direction: Unit directions in which the gates are passed, with shape (n, 3).
        overshoot: Distance in meters of the last knot past the last gate.

    Returns:
        - The knots with shape (n + 2, 3).
        - The unit tangents with shape (n + 2, 3): towards the first gate at the start, then the
          gates' directions.
    """
    to_first_gate = gates_pos[0] - start
    knots = np.vstack((start, gates_pos, gates_pos[-1] + overshoot * gates_direction[-1]))
    tangents = np.vstack(
        (to_first_gate / la.norm(to_first_gate), gates_direction, gates_direction[-1])
    )
    return knots, tangents


class CenterlinePlanner:
    """Plans the centerline from the start position through the gates in their sequence.

    When a gate's true pose is revealed, the whole centerline is rebuilt from the same start, so
    the tunnel always describes the track and not the drone's current state.
    """

    # The centerline always starts at the drone's initial position, so the progress is projected
    # onto a new centerline instead of restarting at zero
    starts_at_drone = False

    def __init__(self, obs: dict[str, NDArray[np.floating]]):
        """Plan the centerline through the gates' poses in the initial observation.

        Args:
            obs: The initial observation of the environment's state.
        """
        self._start = np.array(obs["pos"], dtype=float)
        self._gate_sequence = np.asarray(obs["gate_sequence"])
        self._gate_sequence_direction = np.asarray(obs["gate_sequence_direction"], dtype=float)
        self._gates_visited = np.array(obs["gates_visited"])
        self._reference = None
        self.gate_progress = None  # Arc length of every gate crossing in meters, shape (n,)
        self._build(obs)

    def plan(self, obs: dict[str, NDArray[np.floating]]) -> Reference:
        """Return the centerline, rebuilt if the observation revealed a gate's true pose.

        Args:
            obs: The current observation of the environment.

        Returns:
            The current centerline.
        """
        if np.any(obs["gates_visited"] & ~self._gates_visited):
            self._gates_visited = np.array(obs["gates_visited"])
            self._build(obs)
        return self._reference

    def _build(self, obs: dict[str, NDArray[np.floating]]):
        """Build the centerline through the gate poses of the observation."""
        gates_pos = np.asarray(obs["gates_pos"])[self._gate_sequence]
        normals = R.from_quat(np.asarray(obs["gates_quat"])[self._gate_sequence]).apply([1, 0, 0])
        directions = normals * self._gate_sequence_direction[:, None]
        knots, tangents = centerline_knots(self._start, gates_pos, directions, OVERSHOOT)
        # Zero second derivatives at the knots, so the centerline is straight through the gates
        accelerations = np.zeros_like(tangents)
        if self._reference is None:
            self._reference = Reference(knots, tangents, accelerations)
        else:
            # Rebuilt in place, which increments the reference's version
            self._reference._update(knots, tangents, accelerations)
        self.gate_progress = self._reference.sparse_s[1:-1]
