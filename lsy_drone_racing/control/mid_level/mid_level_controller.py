"""Mid Level Controller generating a trajectory from a geometrical path.

The trajectory is parameterized by positions and velocity
"""

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray

from lsy_drone_racing.control.planner.high_level_planner import SplinePlanner
from lsy_drone_racing.control.planner.reference import ProgressReference
from lsy_drone_racing.control.utils.types import ObservationDict, Trajectory


class MidLevelController:
    def __init__(
        self,
        planner: SplinePlanner,
        v_max: float,
        a_long_max: float,
        a_lat_max: float,
        a_brake_max: float,
        v_gate_max: float = None,
        gate_window: float = 0.5,
        gate_plateau: float = 0.0,
        progress_window: tuple[float, float] = (0.2, 0.2),
    ):

        self._planner = planner

        self.v_max = v_max
        # speed cap at a gate's progress, rising linearly back to v_max gate_window [m] progress away
        self.v_gate_max = v_max if v_gate_max is None else v_gate_max
        self.gate_window = gate_window
        # the cap is flat within gate_plateau [m] of a gate's progress, so it holds until the gate is revealed
        self.gate_plateau = gate_plateau
        self.a_long_max = a_long_max
        self.a_lat_max = a_lat_max
        self.a_brake_max = a_brake_max
        # (behind, ahead) [m] window around the predicted progress in which the path is searched
        self.progress_window = progress_window

        self._current_velocity_profile = None

        self._current_progress = 0.0

    def get_trajectory(self, obs: ObservationDict, horizon: int, dt: float) -> Trajectory:
        """Compute trajectory of horizon H for current reference path."""
        ref, updated = self._planner.plan(obs)

        if updated:
            self._current_velocity_profile = self._compute_velocity_profile(ref, obs)

        s_k = np.zeros((horizon + 1,))
        v_k = np.zeros((horizon + 1,))

        current_speed = la.norm(obs["vel"])
        # predict progress by integrating the velocity component along the path tangent, then
        # correct it by projecting onto the path in a small window around the prediction
        tangent = ref(self._current_progress, 1)
        tangent = tangent / max(la.norm(tangent), 1e-9)
        s_pred = np.clip(self._current_progress + dt * np.dot(obs["vel"], tangent), 0.0, ref.length)
        current_progress = ref.project(obs["pos"], s_pred=s_pred)

        s_k[0] = current_progress
        v_k[0] = current_speed

        def _velocity_profile(progress: float) -> NDArray:

            i = np.clip(np.searchsorted(ref.dense_s, progress), 0, len(ref.dense_s) - 1)

            return self._current_velocity_profile[i]

        for k in range(horizon):
            v_k[k + 1] = min(_velocity_profile(s_k[k]), v_k[k] + self.a_long_max * dt)
            s_k[k + 1] = s_k[k] + v_k[k] * dt  # ZOH

        ref_window = np.hstack([ref(s_k), ref(s_k, 1) * v_k[:, None]])

        self._current_progress = current_progress
        return ref_window  # (H + 1, 6)

    def _compute_velocity_profile(
        self, ref: ProgressReference, obs: ObservationDict, eps: float = 1e-6, pass_tol: float = 0.2
    ) -> NDArray:

        # our path is parameterized by arc length
        s = ref.dense_s
        # therefore the second derivative is curvature kappa
        kappa = la.norm(ref(s, 2), axis=-1)

        # kappa: 1 / R
        # a_lat = v^2 / R
        # => v = sqrt(a_lat / kappa)
        v = np.minimum(self.v_max, np.sqrt(np.maximum(self.a_lat_max / kappa, eps)))

        # slow down around the first pass of each gate whose true position has not been revealed yet
        d = la.norm(ref(s)[:, None, :] - obs["gates_pos"][None, :, :], axis=-1)  # (N, G)
        # gate passes are local distance minima within the gate opening
        is_pass = np.zeros_like(d, dtype=bool)
        is_pass[1:-1] = (d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < pass_tol)

        slow_down = is_pass.any(axis=0) & ~obs["gates_visited"]
        s_gate = s[is_pass.argmax(axis=0)][slow_down]  # progress of the first pass of each gate

        closeness = np.clip(
            (np.abs(s[:, None] - s_gate[None, :]) - self.gate_plateau) / self.gate_window, 0.0, 1.0
        )
        closeness = closeness.min(axis=1, initial=1.0)
        v = np.minimum(v, self.v_gate_max + (self.v_max - self.v_gate_max) * closeness)

        ds = np.diff(s)
        for i in range(len(v) - 1):  # cap at max acceleration
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * self.a_long_max * ds[i]))

        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self.a_brake_max * ds[i]))

        return v
