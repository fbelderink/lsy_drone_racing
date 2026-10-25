"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The reference path is built from the gates in the observation: for each gate in the race sequence
we place a point before, at, and after the gate along its passing direction and fit a cubic spline
through these points, parameterized by arc length. Where the spline comes too close to an obstacle
or a gate frame, detour points are added and the spline is fitted again. The MPC reference moves
along this path following a speed profile that slows down in tight curves. Whenever the true pose
of a gate or obstacle is revealed, the path is rebuilt from the drone's current position through
the remaining gates.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

GATE_OPENING_HALF_SIZE = 0.2  # m, the opening is 0.4 m x 0.4 m
GATE_FRAME_HALF_SIZE = 0.36  # m, the outer frame is 0.72 m x 0.72 m
# Detour points are placed this much further out than required. Otherwise the refitted spline dips
# slightly back inside between the detour point and its neighbors and the detours never converge.
DETOUR_EXTRA_DISTANCE = 0.05  # m


def max_horizontal_randomization(config: dict, name: str) -> float:
    """Largest horizontal shift in m of the `name` randomization in the config, 0 if not used."""
    randomizations = config.env.get("randomizations") or {}
    if randomizations.get(name) is None:
        return 0.0
    kwargs = randomizations[name]["kwargs"]
    return float(np.max(np.abs([kwargs["minval"][:2], kwargs["maxval"][:2]])))


def obstacle_detours(
    path_points: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    clearances: NDArray[np.floating],
) -> list[tuple[int, NDArray[np.floating]]]:
    """Find all obstacles that the path passes closer than their clearance.

    Args:
        path_points: Dense samples of the path, shape (n_samples, 3).
        obstacles_pos: Positions of the obstacles, shape (n_obstacles, 3).
        clearances: Required horizontal distance in m from each obstacle, shape (n_obstacles,).

    Returns:
        For each such obstacle, the index of the closest path sample and the detour points, shape
        (1, 3): the sample moved sideways, away from the obstacle.
    """
    detours = []
    for obstacle_pos, clearance in zip(obstacles_pos, clearances):
        offsets = path_points[:, :2] - obstacle_pos[:2]  # Obstacles are poles, so only use x, y
        distances = np.linalg.norm(offsets, axis=-1)
        sample = int(np.argmin(distances))
        if distances[sample] >= clearance:
            continue
        away = offsets[sample] / max(distances[sample], 1e-6)
        detour = path_points[sample].copy()
        detour[:2] = obstacle_pos[:2] + (clearance + DETOUR_EXTRA_DISTANCE) * away
        detours.append((sample, detour[None]))
    return detours


def gate_frame_detours(
    path_points: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_quat: NDArray[np.floating],
    margins: NDArray[np.floating],
) -> list[tuple[int, NDArray[np.floating]]]:
    """Find all gates whose frame or stand the path comes too close to.

    Only samples closer than the gate's margin to its plane can hit the frame. Those must either
    be inside the opening (shrunk by half the margin) or outside the frame (grown by the margin).

    Args:
        path_points: Dense samples of the path, shape (n_samples, 3).
        gates_pos: Positions of all gates, shape (n_gates, 3).
        gates_quat: Orientations of all gates as xyzw quaternions, shape (n_gates, 4).
        margins: Required distance in m from each gate's frame, shape (n_gates,).

    Returns:
        For each such gate, the index of the first offending path sample and the detour points,
        shape (k, 3):

        * If the path crosses the gate's plane there, two points, one on each side of the plane,
          both beside or above the frame (whichever is the smaller move). The path then crosses
          the plane cleanly outside the frame.
        * If the path only comes close to the plane and returns to the same side, one point that
          moves the sample away from the plane.
    """
    detours = []
    for gate_pos, gate_quat, margin in zip(gates_pos, gates_quat, margins):
        inner = GATE_OPENING_HALF_SIZE - margin / 2
        outer = GATE_FRAME_HALF_SIZE + margin
        gate_rot = R.from_quat(gate_quat)
        x, y, z = gate_rot.apply(path_points - gate_pos, inverse=True).T  # x: through the gate
        near_plane = np.abs(x) < margin
        through_opening = (np.abs(y) < inner) & (np.abs(z) < inner)
        hits_frame_or_stand = (np.abs(y) < outer) & (z < outer)  # The stand reaches the ground
        bad = np.nonzero(near_plane & ~through_opening & hits_frame_or_stand)[0]
        if len(bad) == 0:
            continue
        sample = int(bad[0])
        # The side of the plane the path comes from and the side it leaves to
        before = np.nonzero(~near_plane[:sample])[0]
        after = np.nonzero(~near_plane[sample:])[0]
        side_in = np.sign(x[before[-1]]) if len(before) else np.sign(x[sample]) or 1.0
        side_out = np.sign(x[sample + after[0]]) if len(after) else side_in
        x_clear = margin + DETOUR_EXTRA_DISTANCE
        if side_in == side_out:
            local_points = [[side_in * x_clear, y[sample], z[sample]]]
        else:
            y_new, z_new = y[sample], z[sample]
            if outer - abs(y_new) < outer - z_new:
                y_new = np.sign(y_new) * (outer + DETOUR_EXTRA_DISTANCE)
            else:
                z_new = outer + DETOUR_EXTRA_DISTANCE
            local_points = [[side_in * x_clear, y_new, z_new], [side_out * x_clear, y_new, z_new]]
        detours.append((sample, gate_pos + gate_rot.apply(local_points)))
    return detours


def build_gate_path(
    start_pos: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_quat: NDArray[np.floating],
    gate_ids: NDArray[np.integer],
    gate_directions: NDArray[np.integer],
    obstacles_pos: NDArray[np.floating],
    gate_offset: float,
    obstacle_clearances: NDArray[np.floating],
    gate_frame_margins: NDArray[np.floating],
) -> CubicSpline:
    """Build a path through the given gates around the obstacles, parameterized by arc length.

    Args:
        start_pos: Start point of the path, usually the drone's current position.
        gates_pos: Positions of all gates, shape (n_gates, 3).
        gates_quat: Orientations of all gates as xyzw quaternions, shape (n_gates, 4).
        gate_ids: Gates to pass, in order (0-based gate IDs).
        gate_directions: Passing direction per entry of `gate_ids`, 1 for forward, -1 for reverse.
        obstacles_pos: Positions of the obstacles, shape (n_obstacles, 3). Obstacles are vertical
            poles, so only their x and y coordinates matter.
        gate_offset: Distance in m of the extra points placed before and after each gate center.
        obstacle_clearances: Minimum horizontal distance in m between the path and each obstacle,
            shape (n_obstacles,).
        gate_frame_margins: Minimum distance in m between the path and the outside of each gate
            frame where the path does not go through the gate opening, shape (n_gates,).

    Returns:
        A spline `path(s)` that returns the position at arc length `s` (in m) along the path.
    """
    # Gates are passed along their local x-axis. Flip it for gates passed in reverse.
    pass_dirs = R.from_quat(gates_quat[gate_ids]).apply([1.0, 0.0, 0.0])
    pass_dirs *= gate_directions[:, None]
    centers = gates_pos[gate_ids]
    # Before, center, after for each gate -> shape (n, 3, 3), in passing order
    gate_points = np.stack(
        [centers - gate_offset * pass_dirs, centers, centers + gate_offset * pass_dirs], axis=1
    )
    # When we replan while flying through the first gate, some of its points are already behind
    # the drone. Keep only those still ahead, otherwise the path would turn back through the gate.
    ahead = (gate_points[0] - start_pos) @ pass_dirs[0] > 0.05
    points = np.vstack([start_pos, gate_points[0][ahead], gate_points[1:].reshape(-1, 3)])

    def fit_and_sample(points: NDArray[np.floating]) -> tuple[NDArray, NDArray, NDArray]:
        """Fit a spline using the straight-line distances between points as parameter `u`."""
        chord_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(chord_lengths)])
        u_dense = np.linspace(0.0, u[-1], 50 * len(points))
        return u, u_dense, CubicSpline(u, points)(u_dense)

    # 1) Fit a first spline. If it hits a gate frame or an obstacle, add detour points there and
    # fit again. Gate frames come first: hitting one is a certain crash, while the obstacle
    # clearance includes a safety margin.
    added_detours = []
    for _ in range(2 * (len(obstacles_pos) + len(gates_pos))):
        u, u_dense, dense_points = fit_and_sample(points)
        candidates = gate_frame_detours(dense_points, gates_pos, gates_quat, gate_frame_margins)
        candidates += obstacle_detours(dense_points, obstacles_pos, obstacle_clearances)
        # If we already added (almost) the same detour, the spline cannot get around this
        # collision, e.g. because a gate's before/after point itself is too close to an obstacle.
        # Accept it and work on the remaining collisions instead.
        candidates = [
            (sample, detour_points)
            for sample, detour_points in candidates
            if all(np.linalg.norm(detour_points[0] - added) > 0.02 for added in added_detours)
        ]
        if not candidates:
            break
        sample, detour_points = candidates[0]
        added_detours.append(detour_points[0])
        insert_at = np.searchsorted(u, u_dense[sample])
        points = np.insert(points, insert_at, detour_points, axis=0)
    _, _, dense_points = fit_and_sample(points)

    # 2) Measure the true length along the curve. Refit with that length as parameter, so that
    # path(s) is exactly s meters along the curve and |path'(s)| = 1.
    segment_lengths = np.linalg.norm(np.diff(dense_points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return CubicSpline(s, dense_points)


def build_speed_profile(
    path: CubicSpline,
    start_speed: float,
    max_speed: float,
    max_lateral_acc: float,
    max_longitudinal_acc: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute how fast to move along an arc-length path: slow in tight curves, fast on straights.

    Args:
        path: Arc-length parameterized path, see `build_gate_path`.
        start_speed: Speed in m/s at the start of the path, usually the drone's current speed.
        max_speed: Upper speed limit in m/s.
        max_lateral_acc: Limit in m/s^2 for the acceleration needed to follow a curve, v^2 / r.
        max_longitudinal_acc: Limit in m/s^2 for speeding up and slowing down along the path.

    Returns:
        Arrays `t`, `s`, `v` of the same length: at time `t` after the start, the reference is at
        arc length `s` and moves with speed `v`.
    """
    s = np.linspace(0.0, path.x[-1], max(int(path.x[-1] / 0.01), 2))
    ds = s[1] - s[0]
    # Curvature of an arc-length parameterized curve is the length of its second derivative
    curvature = np.linalg.norm(path.derivative(2)(s), axis=1)
    v = np.minimum(max_speed, np.sqrt(max_lateral_acc / np.maximum(curvature, 1e-6)))
    v[0] = min(v[0], start_speed)
    # Forward pass limits speeding up, backward pass makes sure we slow down before tight curves
    for i in range(1, len(s)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * max_longitudinal_acc * ds))
    for i in range(len(s) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * max_longitudinal_acc * ds))
    # Time to traverse each step, using the average speed of both ends
    t = np.concatenate([[0.0], np.cumsum(2 * ds / np.maximum(v[:-1] + v[1:], 1e-3))])
    return t, s, v


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the symbolic drone dynamics."""
    # For more info on the models, check out https://github.com/learnsyslab/crazyflow
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            50.0,  # pos
            50.0,  # pos
            400.0,  # pos
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            30.0,  # rpy
            30.0,  # rpy
            30.0,  # rpy
            40.0,  # thrust
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    total_thrust_min = parameters["thrust_min"] * 4
    total_thrust_max = parameters["thrust_max"] * 4
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, total_thrust_min])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, total_thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    cmake_builder = None
    # On Windows, use CMake with MinGW to build the generated MPC solver DLL.
    if os.name == "nt":
        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.build_dir = "build-mingw"

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
        cmake_builder=cmake_builder,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        self._max_speed = 1.5  # m/s, speed of the reference on straight parts of the path
        self._max_lateral_acc = 2.0  # m/s^2, limits the speed in curves
        self._max_longitudinal_acc = 2.0  # m/s^2, limits speeding up and slowing down
        self._gate_offset = 0.2  # m, distance of the points before/after each gate center
        # Must stay below the distance of the before/after points to the obstacles, otherwise the
        # path cannot be fixed. On the default track, an obstacle stands 0.5 m behind a gate.
        self._obstacle_clearance = 0.25  # m, minimum horizontal distance of the path to obstacles
        self._gate_frame_margin = 0.15  # m, minimum distance of the path to the outside of frames
        # Until a gate or obstacle is within sensor range, we only know its nominal position. The
        # true one can be off by up to the randomization range, so keep that much extra distance.
        self._unseen_gate_extra = max_horizontal_randomization(config, "gate_pos")
        self._unseen_obstacle_extra = max_horizontal_randomization(config, "obstacle_pos")
        self._tick = 0
        self._replan(obs)

        self.drone_params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._config = config

    def _replan(self, obs: dict[str, NDArray[np.floating]]):
        """Build a new path from the drone's position through all gates not yet passed."""
        n_passed = int(obs["n_gates_passed"])
        self._path = build_gate_path(
            obs["pos"],
            obs["gates_pos"],
            obs["gates_quat"],
            obs["gate_sequence"][n_passed:],
            obs["gate_sequence_direction"][n_passed:],
            obs["obstacles_pos"],
            self._gate_offset,
            self._obstacle_clearance + self._unseen_obstacle_extra * ~obs["obstacles_visited"],
            self._gate_frame_margin + self._unseen_gate_extra * ~obs["gates_visited"],
        )
        self._path_length = self._path.x[-1]
        self._path_tangent = self._path.derivative()
        self._profile_t, self._profile_s, self._profile_v = build_speed_profile(
            self._path,
            float(np.linalg.norm(obs["vel"])),
            self._max_speed,
            self._max_lateral_acc,
            self._max_longitudinal_acc,
        )
        self._path_start_tick = self._tick  # The reference starts at s = 0 at this tick
        self._gates_visited = obs["gates_visited"].copy()
        self._obstacles_visited = obs["obstacles_visited"].copy()

    def _reference(
        self, ticks: NDArray[np.integer]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Reference positions and velocities at the given ticks.

        The reference moves along the path following the speed profile and stops at the end.
        """
        t = (ticks - self._path_start_tick) * self._dt
        s = np.interp(t, self._profile_t, self._profile_s)  # Stays at the end after the last time
        speed = np.interp(t, self._profile_t, self._profile_v)
        speed[t >= self._profile_t[-1]] = 0.0
        pos = self._path(s)
        vel = speed[:, None] * self._path_tangent(s)  # Tangent has unit length
        return pos, vel

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        # Rebuild the path as soon as the true pose of another gate or obstacle becomes known
        newly_seen_gate = np.any(obs["gates_visited"] & ~self._gates_visited)
        newly_seen_obstacle = np.any(obs["obstacles_visited"] & ~self._obstacles_visited)
        if newly_seen_gate or newly_seen_obstacle:
            self._replan(obs)
        ref_pos, ref_vel = self._reference(self._tick + np.arange(self._N + 1))

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Setting state reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = ref_pos[: self._N]  # position
        # zero roll, pitch, yaw
        yref[:, 6:9] = ref_vel[: self._N]  # velocity
        # zero drpy

        # Setting input reference (index > self._nx)
        # zero rpy
        # hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = ref_pos[self._N]  # position
        # zero roll, pitch, yaw
        yref_e[6:9] = ref_vel[self._N]  # velocity
        # zero drpy
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            Always False. The environment ends the episode after the last gate or on a crash.
        """
        self._tick += 1
        return False

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Draw the planned path (green) and the MPC reference over the horizon (red)."""
        path = self._path(np.linspace(0.0, self._path_length, 200))
        draw_line(sim, path, rgba=(0.0, 1.0, 0.0, 1.0))
        horizon, _ = self._reference(self._tick + np.arange(self._N + 1))
        draw_points(sim, horizon, rgba=(1.0, 0.0, 0.0, 1.0), size=0.01)
