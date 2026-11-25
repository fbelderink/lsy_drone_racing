"""Mid Level Controller generating a trajectory from a geometrical path.

The trajectory is parameterized by positions and velocity
"""

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray

from lsy_drone_racing.control.mid_level.mlc import MidLevelController
from lsy_drone_racing.control.planner.high_level_planner import SplinePlanner
from lsy_drone_racing.control.planner.reference import ProgressReference
from lsy_drone_racing.control.utils.types import ObservationDict, Trajectory


class CurvatureMidLevelController(MidLevelController):
    """Times the reference path with curvature and acceleration limits applied separately.

    The speed profile is limited by the lateral acceleration in curves, v = sqrt(a_lat_max / kappa),
    by v_max and by a cap around unrevealed gates. A forward pass then limits speeding up to
    a_long_max and a backward pass braking to a_brake_max.
    """

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
    ):
        """Initialize the controller.

        Args:
            planner: The high level planner that provides the reference path.
            v_max: Speed cap in m/s.
            a_long_max: Largest acceleration in m/s^2 along the path when speeding up.
            a_lat_max: Largest lateral acceleration in m/s^2 in curves.
            a_brake_max: Largest deceleration in m/s^2 along the path when braking.
            v_gate_max: Speed cap in m/s at the progress of a gate whose true position is not
                revealed yet, rising linearly back to v_max over gate_window. v_max if None.
            gate_window: Progress in meters over which the gate cap rises back to v_max.
            gate_plateau: Progress in meters around a gate over which the gate cap stays flat, so
                it holds until the gate is revealed.
        """
        self._planner = planner

        self.v_max = v_max
        self.a_long_max = a_long_max
        self.a_lat_max = a_lat_max
        self.a_brake_max = a_brake_max
        self.v_gate_max = v_max if v_gate_max is None else v_gate_max
        self.gate_window = gate_window
        self.gate_plateau = gate_plateau

        self._reference: ProgressReference | None = None
        self._grid: NDArray | None = None
        self._profile: NDArray | None = None
        self._progress: float = 0.0

    @property
    def reference(self) -> ProgressReference:
        """The reference path the trajectory follows."""
        return self._reference

    @property
    def progress(self) -> float:
        """The drone's progress s in meters on the reference path."""
        return self._progress

    def speed_limit(self, s: NDArray) -> NDArray:
        """Speed of the velocity profile at progress s."""
        return np.interp(s, self._grid, self._profile)

    def get_trajectory(self, obs: ObservationDict, horizon: int, dt: float) -> Trajectory:
        """Compute trajectory of horizon H for current reference path."""
        ref, updated = self._planner.plan(obs)

        if updated or self._reference is None:
            self._reference = ref
            self._grid, self._profile = self._compute_velocity_profile(ref, obs)

        s_k = np.zeros((horizon + 1,))
        v_k = np.zeros((horizon + 1,))

        current_speed = la.norm(obs["vel"])
        # predict progress by integrating the velocity component along the path tangent, then
        # correct it by projecting onto the path in a small window around the prediction
        tangent = ref(self._progress, 1)
        tangent = tangent / max(la.norm(tangent), 1e-9)
        s_pred = np.clip(self._progress + dt * np.dot(obs["vel"], tangent), 0.0, ref.length)
        current_progress = ref.project(obs["pos"], s_pred=s_pred)

        s_k[0] = current_progress
        v_k[0] = current_speed

        for k in range(horizon):
            v_k[k + 1] = min(self.speed_limit(s_k[k]), v_k[k] + self.a_long_max * dt)
            s_k[k + 1] = s_k[k] + v_k[k] * dt  # ZOH

        ref_window = np.hstack([ref(s_k), ref(s_k, 1) * v_k[:, None]])

        self._progress = current_progress
        return ref_window  # (H + 1, 6)

    def _compute_velocity_profile(
        self, ref: ProgressReference, obs: ObservationDict, eps: float = 1e-6, pass_tol: float = 0.2
    ) -> tuple[NDArray, NDArray]:

        # our path is parameterized by arc length
        grid = ref.dense_s
        # therefore the second derivative is curvature kappa
        kappa = la.norm(ref(grid, 2), axis=-1)

        # kappa: 1 / R
        # a_lat = v^2 / R
        # => v = sqrt(a_lat / kappa)
        v = np.minimum(self.v_max, np.sqrt(np.maximum(self.a_lat_max / kappa, eps)))

        # slow down around the first pass of each gate whose true position has not been revealed yet
        d = la.norm(ref(grid)[:, None, :] - obs["gates_pos"][None, :, :], axis=-1)  # (N, G)
        # gate passes are local distance minima within the gate opening
        is_pass = np.zeros_like(d, dtype=bool)
        is_pass[1:-1] = (d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < pass_tol)

        slow_down = is_pass.any(axis=0) & ~obs["gates_visited"]
        s_gate = grid[is_pass.argmax(axis=0)][slow_down]  # progress of the first pass of each gate

        closeness = np.clip(
            (np.abs(grid[:, None] - s_gate[None, :]) - self.gate_plateau) / self.gate_window,
            0.0,
            1.0,
        )
        closeness = closeness.min(axis=1, initial=1.0)
        v = np.minimum(v, self.v_gate_max + (self.v_max - self.v_gate_max) * closeness)

        ds = np.diff(grid)
        for i in range(len(v) - 1):  # cap at max acceleration
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * self.a_long_max * ds[i]))

        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self.a_brake_max * ds[i]))

        return grid, v
