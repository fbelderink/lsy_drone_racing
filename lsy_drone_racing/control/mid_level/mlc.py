"""Interface between the mid level controllers and the low level controllers that follow them.

A mid level controller turns the high level planner's geometric reference path into a reference
trajectory: it decides how fast to fly along the path. Low level controllers only use the members
of `AbstractMidLevelController`, so mid level controllers can be swapped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.planner.reference import ProgressReference
    from lsy_drone_racing.control.utils.types import ObservationDict, Trajectory


class MidLevelController(ABC):
    """Times the high level planner's reference path for a low level controller."""

    @abstractmethod
    def get_trajectory(self, obs: ObservationDict, horizon: int, dt: float) -> Trajectory:
        """Reference trajectory from the drone's current progress on.

        Args:
            obs: The current observation, passed on to the high level planner.
            horizon: Number of steps H of the trajectory.
            dt: Time step in seconds between two points of the trajectory.

        Returns:
            (H + 1, 6) positions and velocities [x, y, z, vx, vy, vz].
        """

    @property
    @abstractmethod
    def reference(self) -> ProgressReference:
        """The reference path the trajectory follows."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def progress(self) -> float:
        """The drone's progress s in meters on the reference path."""
        raise NotImplementedError()

    @abstractmethod
    def speed_limit(self, s: NDArray) -> NDArray:
        """Speed in m/s the drone may fly at progress s (any shape) on the reference path."""
        raise NotImplementedError()


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
