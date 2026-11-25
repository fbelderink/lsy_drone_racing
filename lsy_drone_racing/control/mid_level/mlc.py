"""Interface between the mid level controllers and the low level controllers that follow them.

A mid level controller turns the high level planner's geometric reference path into a reference
trajectory: on top of tracking the path and the drone's progress (`PathTracker`), it decides how
fast to fly along the path. Low level controllers only use the members of `MidLevelController`,
so mid level controllers can be swapped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la

from lsy_drone_racing.control.mid_level.path_tracker import PathTracker

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.planner.high_level_planner import SplinePlanner
    from lsy_drone_racing.control.planner.reference import ProgressReference
    from lsy_drone_racing.control.utils.types import ObservationDict, Trajectory


class MidLevelController(PathTracker, ABC):
    """Times the high level planner's reference path for a low level controller.

    Subclasses define the speed along the path with `speed_limit`, usually from a profile they
    compute in `_on_new_reference` whenever the planner provides a new path.
    """

    def __init__(self, planner: SplinePlanner, a_long_max: float):
        """Initialize the controller.

        Args:
            planner: The high level planner that provides the reference path.
            a_long_max: Acceleration in m/s^2 with which the trajectory catches up with the speed
                limit when the drone is slower.
        """
        super().__init__(planner)
        self.a_long_max = a_long_max

    @abstractmethod
    def speed_limit(self, s: NDArray) -> NDArray:
        """Speed in m/s the drone may fly at progress s (any shape) on the reference path."""

    @abstractmethod
    def _on_new_reference(self, ref: ProgressReference, obs: ObservationDict):
        """Compute what `speed_limit` needs for a new path."""

    def get_trajectory(self, obs: ObservationDict, horizon: int, dt: float) -> Trajectory:
        """Reference trajectory from the drone's current progress on.

        Starting at the drone's speed, the trajectory speeds up with at most a_long_max towards
        the speed limit.

        Args:
            obs: The current observation, passed on to the high level planner.
            horizon: Number of steps H of the trajectory.
            dt: Time step in seconds between two points of the trajectory.

        Returns:
            (H + 1, 6) positions and velocities [x, y, z, vx, vy, vz].
        """
        self.update(obs, horizon, dt)
        ref = self._reference

        s = np.zeros(horizon + 1)
        v = np.zeros(horizon + 1)
        s[0], v[0] = self._progress, la.norm(obs["vel"])
        for k in range(horizon):
            v[k + 1] = min(self.speed_limit(s[k]), v[k] + self.a_long_max * dt)
            s[k + 1] = s[k] + v[k] * dt  # ZOH

        return np.hstack([ref(s), ref(s, nu=1) * v[:, None]])


def gate_speed_cap(
    ref: ProgressReference,
    obs: ObservationDict,
    s: NDArray,
    v_max: float,
    v_gate_max: float,
    gate_plateau: float,
    gate_window: float,
    pass_tol: float = 0.2,
) -> NDArray:
    """Speed cap (N,) at progress s (N,): v_max, lowered around the unrevealed gates.

    Around the first pass of each gate whose true position has not been revealed yet, the cap is
    v_gate_max within gate_plateau of the gate's progress and rises linearly back to v_max over
    gate_window.

    Args:
        ref: The reference path.
        obs: The current observation.
        s: (N,) increasing progress values, dense enough to find the gate passes.
        v_max: Speed cap away from unrevealed gates.
        v_gate_max: Speed cap at an unrevealed gate.
        gate_plateau: Progress in meters around a gate over which the cap stays v_gate_max.
        gate_window: Progress in meters over which the cap rises back to v_max.
        pass_tol: Largest distance in meters to a gate's center that counts as a pass.
    """
    d = la.norm(ref(s)[:, None, :] - obs["gates_pos"][None, :, :], axis=-1)  # (N, G)
    # gate passes are local distance minima within the gate opening
    is_pass = np.zeros_like(d, dtype=bool)
    is_pass[1:-1] = (d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < pass_tol)

    slow_down = is_pass.any(axis=0) & ~obs["gates_visited"]
    s_gate = s[is_pass.argmax(axis=0)][slow_down]  # progress of the first pass of each gate

    closeness = np.clip(
        (np.abs(s[:, None] - s_gate[None, :]) - gate_plateau) / gate_window, 0.0, 1.0
    )
    closeness = closeness.min(axis=1, initial=1.0)
    return v_gate_max + (v_max - v_gate_max) * closeness
