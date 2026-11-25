"""Mid level controller that times the high level reference with TOPP-RA.

TOPP-RA (time-optimal path parameterization by reachability analysis) finds the fastest speed
profile along a fixed path under constraints that are linear in the path acceleration s'' and the
squared path speed s'^2. With the path parameterized by arc length s, the drone's acceleration is
q'(s) s'' + q''(s) s'^2, so the mass-specific thrust it needs, acceleration plus gravity, is such
a linear term. Its limits (thrust range and tilt) are approximated by a polytope, which couples
braking, speeding up and turning: the drone can turn less sharply while it accelerates.

The speed profile is computed once per reference: the backward pass gives the speeds from which
the rest of the path stays feasible, the forward pass then speeds up as fast as the thrust allows
within them. The trajectory rollout follows that profile from the drone's current speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.linalg as la
import toppra as ta
from crazyflow.dynamics import load_params
from toppra.constraint import LinearConstraint
from toppra.interpolator import AbstractGeometricPath

from lsy_drone_racing.control.mid_level.mlc import MidLevelController, gate_speed_cap

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.planner.high_level_planner import SplinePlanner
    from lsy_drone_racing.control.planner.reference import ProgressReference
    from lsy_drone_racing.control.utils.types import ObservationDict, Trajectory


def thrust_limits(drone: str) -> tuple[float, float, float]:
    """Mass-specific minimum and maximum total thrust of the drone, and gravity, in m/s^2."""
    params = load_params("so_rpy", drone)
    mass, gravity = float(params["mass"]), float(-params["gravity_vec"][2])
    return 4 * float(params["thrust_min"]) / mass, 4 * float(params["thrust_max"]) / mass, gravity


class ReferencePath(AbstractGeometricPath):
    """A `ProgressReference` as a TOPP-RA path, parameterized by arc length."""

    def __init__(self, ref: ProgressReference):
        """Wrap the reference path."""
        self._ref = ref

    def __call__(self, path_positions: float | NDArray, order: int = 0) -> NDArray:
        """Position, unit tangent (order 1) or curvature vector (order 2) at the progress values."""
        return self._ref(path_positions, nu=order)

    @property
    def dof(self) -> int:
        """Dimension of the path."""
        return 3

    @property
    def path_interval(self) -> NDArray:
        """Start and end progress."""
        return np.array([0.0, self._ref.length])


class ThrustConstraint(LinearConstraint):
    """Limits on the mass-specific thrust v = acceleration + gravity, and a speed cap.

    v = q'(s) s'' + q''(s) s'^2 + g e_z has to stay within tilt_max of the vertical, below
    thrust_max and above thrust_min. The cone and the sphere are approximated by polytopes inside
    them, so the polytope is conservative. The speed cap bounds s'^2 per grid point.
    """

    def __init__(
        self,
        thrust_min: float,
        thrust_max: float,
        tilt_max: float,
        gravity: float,
        speed_cap: Callable[[NDArray], NDArray],
        n_azimuth: int = 16,
        n_rings: int = 4,
    ):
        """Build the polytope.

        Args:
            thrust_min: Minimum mass-specific thrust in m/s^2.
            thrust_max: Maximum mass-specific thrust in m/s^2.
            tilt_max: Largest angle in rad between the thrust and the vertical.
            gravity: Gravitational acceleration in m/s^2.
            speed_cap: Maps progress values (N,) to the speed cap (N,) in m/s there.
            n_azimuth: Number of facets around the vertical.
            n_rings: Number of facet rings between the vertical and tilt_max for the thrust limit.
        """
        super().__init__()
        self.identical = True  # the same polytope at every grid point
        self._gravity = np.array([0.0, 0.0, gravity])
        self._speed_cap = speed_cap

        phi = 2 * np.pi * np.arange(n_azimuth) / n_azimuth
        # tilt: horizontal thrust <= tan(tilt_max) * vertical thrust, a polygon inside the cone
        tilt = np.column_stack(
            [
                np.cos(phi),
                np.sin(phi),
                np.full(n_azimuth, -np.tan(tilt_max) * np.cos(np.pi / n_azimuth)),
            ]
        )
        # thrust: facets with normals spread over the cone, moved inwards by the largest half gap
        # between them, so that the polytope lies inside the sphere of radius thrust_max
        alpha = np.linspace(0.0, tilt_max, n_rings)[1:]
        a, p = np.meshgrid(alpha, phi)
        directions = np.vstack(
            [
                [0.0, 0.0, 1.0],
                np.column_stack(
                    [
                        np.sin(a.ravel()) * np.cos(p.ravel()),
                        np.sin(a.ravel()) * np.sin(p.ravel()),
                        np.cos(a.ravel()),
                    ]
                ),
            ]
        )
        half_gap = max(tilt_max / (n_rings - 1), np.pi / n_azimuth * np.sin(tilt_max)) / 2
        # minimum thrust: vertical thrust >= thrust_min
        self._F = np.vstack([tilt, directions, [0.0, 0.0, -1.0]])
        self._g = np.concatenate(
            [
                np.zeros(n_azimuth),
                np.full(len(directions), thrust_max * np.cos(half_gap)),
                [-thrust_min],
            ]
        )

    def compute_constraint_params(
        self, path: AbstractGeometricPath, gridpoints: NDArray, *args: object, **kwargs: object
    ) -> tuple:
        """Coefficients of v = a s'' + b s'^2 + c, F v <= g, and the bounds on s'^2."""
        a, b = path(gridpoints, 1), path(gridpoints, 2)
        c = np.tile(self._gravity, (len(gridpoints), 1))
        xbound = np.column_stack([np.zeros(len(gridpoints)), self._speed_cap(gridpoints) ** 2])
        return a, b, c, self._F, self._g, None, xbound


class TOPPRAMidLevelController(MidLevelController):
    """Times the reference path with the fastest speed profile the drone's thrust allows."""

    def __init__(
        self,
        planner: SplinePlanner,
        config: dict,
        v_max: float,
        a_long_max: float,
        v_gate_max: float | None = None,
        gate_window: float = 0.5,
        gate_plateau: float = 0.0,
        thrust_ratio: float = 0.8,
        tilt_max: float = 1.0,
        grid_ds: float = 0.01,
    ):
        """Initialize the controller.

        Args:
            planner: The high level planner that provides the reference path.
            config: Config dictionary.
            v_max: Speed cap in m/s.
            a_long_max: Acceleration in m/s^2 with which the rollout catches up with the profile
                when the drone is slower than it.
            v_gate_max: Speed cap at gates whose true position is not revealed yet, v_max if None.
            gate_window: Progress in meters over which the gate cap rises back to v_max.
            gate_plateau: Progress in meters around a gate over which the gate cap holds.
            thrust_ratio: Share of the maximum thrust the profile may use, the rest is left to
                the low level controller for corrections.
            tilt_max: Largest tilt in rad of the thrust from the vertical.
            grid_ds: Spacing in meters of TOPP-RA's grid; has to resolve the sharpest curves.
        """
        self._planner = planner
        self.v_max = v_max
        self.a_long_max = a_long_max
        self.v_gate_max = v_max if v_gate_max is None else v_gate_max
        self.gate_window = gate_window
        self.gate_plateau = gate_plateau
        thrust_min, thrust_max, self._gravity = thrust_limits(config.sim.drone)
        self._thrust = (thrust_min, thrust_ratio * thrust_max)
        self.tilt_max = tilt_max
        self.grid_ds = grid_ds

        self._reference: ProgressReference | None = None
        self._grid: NDArray | None = None  # (N,) progress of TOPP-RA's grid points
        self._profile: NDArray | None = None  # (N,) fastest speed at the grid points
        self._progress = 0.0

    @property
    def reference(self) -> ProgressReference:
        """The reference path the trajectory follows."""
        return self._reference

    @property
    def progress(self) -> float:
        """The drone's progress s in meters on the reference path."""
        return self._progress

    def speed_limit(self, s: NDArray) -> NDArray:
        """Speed of the TOPP-RA profile at progress s (any shape)."""
        return np.interp(s, self._grid, self._profile)

    def get_trajectory(self, obs: ObservationDict, horizon: int, dt: float) -> Trajectory:
        """Reference trajectory from the drone's current progress on, following the profile."""
        ref, updated = self._planner.plan(obs)
        if updated or self._reference is None:
            self._reference = ref
            self._grid, self._profile = self._time_optimal_profile(ref, obs)

        # predict the progress by integrating the speed along the path, then correct it by
        # projecting the position onto the path around the prediction
        tangent = ref(self._progress, nu=1)
        s_pred = np.clip(self._progress + dt * np.dot(obs["vel"], tangent), 0.0, ref.length)
        self._progress = float(ref.project(obs["pos"], s_pred=s_pred))

        s = np.zeros(horizon + 1)
        v = np.zeros(horizon + 1)
        s[0], v[0] = self._progress, la.norm(obs["vel"])
        for k in range(horizon):
            v[k + 1] = min(self.speed_limit(s[k]), v[k] + self.a_long_max * dt)
            s[k + 1] = s[k] + v[k] * dt

        return np.hstack([ref(s), ref(s, nu=1) * v[:, None]])  # (H + 1, 6)

    def _time_optimal_profile(
        self, ref: ProgressReference, obs: ObservationDict, max_retries: int = 100
    ) -> tuple[NDArray, NDArray]:
        """Fastest speed profile along the whole reference, on TOPP-RA's grid.

        The end speed is free: the backward pass starts from any speed up to the cap at the end.
        The forward pass starts as fast as feasible, the rollout limits the speed-up from the
        drone's actual speed.

        Returns:
            grid (N,) progress and speed (N,) at the grid points.
        """
        grid = np.linspace(0.0, ref.length, int(np.ceil(ref.length / self.grid_ds)) + 1)
        cap = gate_speed_cap(
            ref, obs, grid, self.v_max, self.v_gate_max, self.gate_plateau, self.gate_window
        )

        def speed_cap(s: NDArray) -> NDArray:
            return np.interp(s, grid, cap)

        constraint = ThrustConstraint(*self._thrust, self.tilt_max, self._gravity, speed_cap)
        toppra = ta.algorithm.TOPPRA(
            [constraint], ReferencePath(ref), gridpoints=grid, solver_wrapper="seidel"
        )

        # backward pass: K[i] are the squared speeds at grid point i from which the rest of the
        # path stays feasible
        K = toppra.compute_controllable_sets(0.0, float(cap[-1]))
        # forward pass: speed up as fast as feasible while staying controllable. TOPP-RA's own
        # forward pass (compute_parameterization) would fix the end speed, so its step is used
        x = np.zeros(len(grid))
        x[0] = K[0, 1]
        deltas = np.diff(grid)
        toppra.solver_wrapper.setup_solver()
        for i in range(len(grid) - 1):
            u = toppra._forward_step(i, x[i], K[i + 1])[0]
            # at the edge of the controllable set the step can be just infeasible numerically;
            # like TOPP-RA's own forward pass, retry slightly slower
            for _ in range(max_retries):
                if not np.isnan(u):
                    break
                x[i] = max(x[i] - 1e-8, 0.999 * x[i])
                u = toppra._forward_step(i, x[i], K[i + 1])[0]
            else:
                if np.isnan(u):
                    raise RuntimeError(f"TOPP-RA forward pass failed at s = {grid[i]:.2f} m")
            x[i + 1] = np.clip(x[i] + 2 * deltas[i] * u, K[i + 1, 0], K[i + 1, 1])
        toppra.solver_wrapper.close_solver()

        return grid, np.sqrt(np.maximum(x, 0.0))
