"""Sampling-based MPC (MPPI) for drone racing using the collective thrust and attitude interface.

The controller plans over the fitted second-order `so_rpy_rotor_drag` dynamics model, i.e. `so_rpy`
with a first-order lag on the collective thrust and a linear body-frame drag. At each step it
samples perturbed control sequences around a nominal sequence, simulates them in parallel, and
updates the nominal sequence with a cost-weighted average of the perturbations.

State and input conventions follow `attitude_mpc.py`, with the thrust state appended:

    x = [pos(3), rpy(3), vel(3), drpy(3), thrust(1)]  (thrust is the lagged collective thrust in N)
    u = [roll, pitch, yaw, thrust]  (thrust is the commanded total collective thrust in N)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy_rotor_drag.dynamics import dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim import visualize
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mid_level.mid_level_controller import MidLevelController
from lsy_drone_racing.control.planner.splines import SplinePlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from jax import Array
    from numpy.typing import NDArray


def dynamics_step(x: Array, u: Array, params: dict, dt: float) -> Array:
    """Advance the so_rpy_rotor_drag model by one time step.

    Args:
        x: State [pos, rpy, vel, drpy, thrust] with shape (13,).
        u: Input [roll, pitch, yaw, thrust] with shape (4,).
        params: Drone parameters as returned by `crazyflow.dynamics.load_params`.
        dt: Integration step in seconds.

    Returns:
        The next state with shape (13,).
    """
    # TODO Check if RK4 integration is necessary, or if Euler is sufficient for the small dt.
    pos, rpy, vel, drpy, thrust = jnp.split(x, [3, 6, 9, 12])
    x_dot = jnp.concatenate(
        dynamics_euler(
            pos=pos,
            rpy=rpy,
            vel=vel,
            rpy_rates=drpy,
            cmd=u,
            rotor_vel=thrust,  # the model stores the lagged thrust in its rotor_vel state
            mass=params["mass"],
            gravity_vec=params["gravity_vec"],
            thrust_dyn_coef=params["thrust_dyn_coef"],
            acc_coef=params["acc_coef"],
            cmd_f_coef=params["cmd_f_coef"],
            cmd_rpy_coef=params["cmd_rpy_coef"],
            rpy_coef=params["rpy_coef"],
            rpy_rates_coef=params["rpy_rates_coef"],
            drag_matrix=params["drag_matrix"],
        )
    )

    # Euler integration step
    x_next = x + dt * x_dot

    return x_next


def rollout(x0: Array, U: Array, params: dict, dt: float) -> Array:
    """Simulate a control sequence starting from x0.

    Args:
        x0: Initial state with shape (13,).
        U: Control sequence with shape (H, 4).
        params: Drone parameters.
        dt: Integration step in seconds.

    Returns:
        The state trajectory with shape (H + 1, 13), including x0.
    """

    # Define a step function for the scan operation
    def step_fn(x, u):
        x_next = dynamics_step(x, u, params, dt)
        return x_next, x_next

    # Use JAX's scan to efficiently compute the trajectory
    _, outputs = jax.lax.scan(step_fn, x0, U)

    # return missing initial state x0 at the beginning of the trajectory
    return jnp.vstack((x0, outputs))


COST_CONSTANTS = {
    "deviation_tolerance": 0.13,  # Tolerance in meters for an acceptable deviation from the reference trajectory
    "velocity_tolerance": 0.5,  # Tolerance in m/s for an acceptable deviation from the reference velocity
    "smoothness_tolerance": 1.0,  # Tolerance for control smoothness in multiples of sigma
    "gate_tolerance": 0.07,  # Tolerance in meters for an acceptable deviation from the gate's center
    "gate_slab": 0.1,  # Half-thickness in meters around a gate's plane in which the gate is checked
    "gate_frame_limit": 0.45,  # Offset in meters from the gate's center beyond which the frame is cleared
    "obstacle_radius": 0.2,  # Radius in meters of the penalty tube around an obstacle
}

COST_WEIGHTS = {
    "deviation_weight": 0.2,  # Weight for deviation from reference trajectory
    "velocity_weight": 0.1,  # Weight for deviation from reference velocity
    "smoothness_weight": 0.01,  # Weight for control smoothness
    "gate_weight": 100.0,  # Weight for gate pass / collision / miss
    "obstacle_weight": 50.0,  # Weight for coming close to an obstacle
}

DEFAULT_PARAMS = {
    "horizon": 30,  # planning steps, 0.6 s at 50 Hz
    "n_samples": 1024,  # sampled control sequences per step
    "sigma": [0.9, 0.9, 0.0, 0.3],  # noise std: roll, pitch, yaw (rad), thrust (N)
    "target_ess": 0.05,  # fraction of samples that should effectively contribute to the update
    "rng_seed": 0,  # seed of the sampling noise
}

MLC_CONSTANTS = {
    "v_max": 6.0,
    "a_long_max": 8.0,  # most important in the current setup for speed
    "a_lat_max": 5.0,
    "a_brake_max": 7.0,
    "v_gate_max": 2.0,  # speed cap within the plateau around a gate
    "gate_plateau": 0.9,  # progress in meters before and after a gate over which the cap stays flat (sensor range plus margin)
    "gate_window": 1.0,  # progress in meters before and after a gate over which the cap applies
}


def trajectory_cost(
    X: Array,
    U: Array,
    u_prev: Array,
    ref: Array,
    gates: Array,
    obstacles_pos: Array,
    sigma: Array,
    cc: dict,
    cw: dict,
) -> Array:
    """Compute the cost of a single rollout.

    Args:
        X: State trajectory with shape (H + 1, 13).
        U: Control sequence with shape (H, 4).
        u_prev: Previous control input with shape (4,).
        ref: Reference positions and tangential velocities with shape (H + 1, 6).
            The first three columns are positions, the last three are velocities.
        gates: Position and normal vector of every gate with shape (n_gates, 6).
            The first three columns are positions, the last three are normal vectors.
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3). Obstacles are vertical
            poles, so only the x and y coordinates are used.
        sigma: Standard deviation of the control input with shape (4,).
        cc: Cost constants, see `COST_CONSTANTS`.
        cw: Cost weights, see `COST_WEIGHTS`.

    Returns:
        The scalar cost.
    """
    # Deviation from reference trajectory
    pos = X[:, :3]  # Extract positions from the state trajectory
    ref_pos = ref[:, :3]  # Extract reference positions
    tol = jnp.maximum(1e-6, cc["deviation_tolerance"])
    pos_error_per_step = jnp.square((pos - ref_pos) / tol)  # Compute per-step position error
    deviation_cost = jnp.sum(pos_error_per_step)

    # Deviation from the reference velocity, so the speed profile (e.g. slowing before gates) is followed
    vel_tol = jnp.maximum(1e-6, cc["velocity_tolerance"])
    vel_error_per_step = jnp.square((X[:, 6:9] - ref[:, 3:6]) / vel_tol)
    velocity_cost = jnp.sum(vel_error_per_step)

    # Cost for control smoothness
    U_including_prev = jnp.vstack(
        (u_prev, U)
    )  # Include previous control input for smoothness calculation
    control_diff = jnp.diff(
        U_including_prev, axis=0
    )  # Compute differences between consecutive control inputs
    control_diff /= cc["smoothness_tolerance"] * jnp.maximum(1e-6, sigma)
    smoothness_cost = jnp.sum(jnp.linalg.norm(control_diff, axis=1))

    # Cost for gate collisi0 on: states near a gate's plane must be close to its center, for every gate
    # get gate_pos and gate_normal
    gate_pos = gates[:, :3]  # Extract gate positions
    gate_normal = gates[:, 3:]  # Extract gate normal vectors
    gate_offset = pos[:, None, :] - gate_pos[None, :, :]  # Offset of each state to each gate center
    # a state is near a gate if it is within a thin slab around the gate's plane
    distance = jnp.sum(gate_offset * gate_normal[None, :, :], axis=-1)
    near_gate_mask = jnp.abs(distance) < cc["gate_slab"]
    # use maxnorm(to not punish corners) of distance to gate center along gate plane primary axes
    # primary axes are always vertical axis and the cross between normal and vertical axis
    vertical_distance = gate_offset[..., 2]
    horizontal_distance = jnp.sum(
        gate_offset * jnp.cross(gate_normal, jnp.array([0, 0, 1]))[None, :, :], axis=-1
    )
    gate_plane_max_distance = jnp.maximum(jnp.abs(vertical_distance), jnp.abs(horizontal_distance))
    # states beside the frame, outside of its outer edge, are free
    frame_mask = near_gate_mask & (gate_plane_max_distance < cc["gate_frame_limit"])

    # cost function based on distance to gate center
    distance_further_than_tolerance = jnp.maximum(
        gate_plane_max_distance / cc["gate_tolerance"] - 1, 0
    )
    gate_cost = jnp.sum(jnp.where(frame_mask, distance_further_than_tolerance**2, 0))

    # Cost for coming close to an obstacle: tube around each pole, checked in the x-y plane
    obstacle_xy_distance = jnp.linalg.norm(pos[:, None, :2] - obstacles_pos[None, :, :2], axis=-1)
    # 0 outside the tube radius, rising to 1 at the pole's axis
    tube_penetration = jnp.maximum(1 - obstacle_xy_distance / cc["obstacle_radius"], 0)
    obstacle_cost = jnp.sum(tube_penetration**2)

    total_cost = (
        cw["deviation_weight"] * deviation_cost
        + cw["velocity_weight"] * velocity_cost
        + cw["smoothness_weight"] * smoothness_cost
        + cw["gate_weight"] * gate_cost
        + cw["obstacle_weight"] * obstacle_cost
    )  # Add other cost components if needed

    return total_cost


def adaptive_lambda(costs: Array, target_ess: float, n_iter: int = 30) -> Array:
    """Find the softmin temperature at which the weights reach a target effective sample size.

    The ESS grows monotonically with the temperature, so it is found by bisection in log-space.

    Args:
        costs: Rollout costs with shape (n_samples,).
        target_ess: Desired effective sample size as a fraction of n_samples, in (0, 1].
        n_iter: Number of bisection steps.

    Returns:
        The temperature lambda.
    """
    costs = costs - jnp.min(costs)
    target = target_ess * costs.shape[0]

    def ess(log_lam):
        weights = jax.nn.softmax(-costs / jnp.exp(log_lam))
        return 1 / jnp.sum(weights**2)

    def bisect(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_peaked = ess(mid) < target
        return jnp.where(too_peaked, mid, lo), jnp.where(too_peaked, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_iter, bisect, (jnp.log(1e-6), jnp.log(1e6)))
    return jnp.exp(0.5 * (lo + hi))


GATE_OPENING_HALF = 0.2  # half width in meters of a gate's opening
N_VIS_BEST = 10  # lowest-cost rollouts that are drawn
N_VIS_RANDOM = 30  # additional evenly strided rollouts that are drawn
VIS_MAX_GEOM = 4000  # geoms the viewer's scene can hold; every line segment and point is one geom


def _free_geoms(sim: Sim) -> int:
    """Return how many more markers fit into the viewer's scene this frame."""
    viewer = getattr(sim.viewer, "viewer", None)
    if viewer is None:
        return 0
    # The scene also holds the model's own geoms, and the markers queued so far for this frame
    return viewer.scn.maxgeom - sim.mj_model.ngeom - len(viewer._markers)


def draw_line(sim: Sim, points: NDArray, **kwargs: Any) -> bool:
    """Draw a line like `crazyflow.sim.visualize.draw_line`, but skip it if the scene is full.

    Returns:
        True if the line was drawn, False if it was skipped.
    """
    if sim.viewer is None or _free_geoms(sim) < len(points) - 1:
        return False
    visualize.draw_line(sim, points, **kwargs)
    return True


def draw_points(sim: Sim, points: NDArray, **kwargs: Any) -> bool:
    """Draw points like `crazyflow.sim.visualize.draw_points`, but skip them if the scene is full.

    Returns:
        True if the points were drawn, False if they were skipped.
    """
    if sim.viewer is None or _free_geoms(sim) < len(points):
        return False
    visualize.draw_points(sim, points, **kwargs)
    return True


def gate_outline(pos: NDArray, quat: NDArray, half: float) -> NDArray:
    """Return the closed square of half width `half` in a gate's plane, shape (5, 3)."""
    corners = np.array(
        [[0, -half, -half], [0, half, -half], [0, half, half], [0, -half, half], [0, -half, -half]]
    )
    return pos + R.from_quat(quat).apply(corners)


def circle_outline(center_xy: NDArray, z: float, radius: float, n: int = 32) -> NDArray:
    """Return a closed horizontal circle around `center_xy` at height `z`, shape (n + 1, 3)."""
    angle = np.linspace(0, 2 * np.pi, n + 1)
    return np.stack(
        (
            center_xy[0] + radius * np.cos(angle),
            center_xy[1] + radius * np.sin(angle),
            np.full(n + 1, z),
        ),
        axis=-1,
    )


def speed_to_rgba(speed: NDArray, alpha: float = 1.0) -> NDArray:
    """Map speeds to colors on a blue (slow) to red (fast) ramp, normalized to the speed range."""
    span = np.ptp(speed)
    s = (speed - speed.min()) / span if span > 0 else np.zeros_like(speed)
    return np.stack((s, 1 - np.abs(2 * s - 1), 1 - s, np.full_like(s, alpha)), axis=-1)


def mppi_update(
    key: Array,
    x0: Array,
    U_nom: Array,
    u_prev: Array,
    ref: Array,
    gates: Array,
    obstacles_pos: Array,
    params: dict,
    dt: float,
    n_samples: int,
    sigma: Array,
    target_ess: float,
    cc: dict,
    cw: dict,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Perform one MPPI iteration.

    Samples `n_samples` perturbed control sequences around `U_nom`, simulates and scores them, and
    returns the nominal sequence updated with the softmin-weighted perturbations. The softmin
    temperature is chosen every step such that the weights reach the target effective sample size.

    Args:
        key: JAX PRNG key.
        x0: Current state with shape (13,).
        U_nom: Nominal control sequence with shape (H, 4).
        u_prev: Previous control sequence with shape (4,).
        ref: Reference positions with shape (H + 1, 3).
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3).
        params: Drone parameters.
        dt: Integration step in seconds.
        n_samples: Number of sampled control sequences.
        sigma: Per-channel standard deviation of the sampling noise with shape (4,).
        target_ess: Desired effective sample size as a fraction of n_samples.
        cc: Cost constants, see `COST_CONSTANTS`.
        cw: Cost weights, see `COST_WEIGHTS`.

    Returns:
        - U_new: The updated nominal control sequence with shape (H, 4).
        - ESS: Effective sample size (ESS) for debugging and logging.
        - lam: The chosen softmin temperature for debugging and logging.
        - vis_pos: Positions of the visualized rollouts with shape (N_VIS_BEST + N_VIS_RANDOM,
          H + 1, 3).
        - vis_weights: Softmin weights of the visualized rollouts with shape
          (N_VIS_BEST + N_VIS_RANDOM,).
        - mean_path: Positions of the rollout of `U_new` with shape (H + 1, 3).

    """
    samples = jax.random.normal(key, (n_samples, len(U_nom), 4), dtype=jnp.float32)
    samples *= sigma
    sampled_actions = U_nom + samples

    # clip attitude and total thrust
    sampled_actions = jnp.clip(
        sampled_actions,
        jnp.array([-1.0, -1.0, -1.0, 4 * params["thrust_min"]]),
        jnp.array([1.0, 1.0, 1.0, 4 * params["thrust_max"]]),
    )

    # define batch functions
    rollout_jit = jax.jit(rollout)
    rollout_batch_of_trajectories = jax.vmap(rollout_jit, in_axes=(None, 0, None, None))

    cost_jit = jax.jit(trajectory_cost)
    trajectory_cost_for_batch = jax.vmap(
        cost_jit, in_axes=(0, 0, None, None, None, None, None, None, None)
    )

    rollouts = rollout_batch_of_trajectories(x0, sampled_actions, params, dt)

    costs = trajectory_cost_for_batch(
        rollouts, sampled_actions, u_prev, ref, gates, obstacles_pos, sigma, cc, cw
    )

    # weighting the runs by costs
    lam = adaptive_lambda(costs, target_ess)
    weights = jax.nn.softmax(-costs / lam)

    # calculate effective sample size (ESS) for debugging and logging
    ess = 1 / jnp.sum(weights**2)

    U_new = jnp.sum(weights[:, None, None] * sampled_actions, axis=0)

    # Rollouts to visualize: the lowest-cost ones plus an evenly strided (i.e. random) subset
    best = jax.lax.top_k(-costs, N_VIS_BEST)[1]
    strided = jnp.arange(N_VIS_RANDOM) * (n_samples // N_VIS_RANDOM)
    vis_idx = jnp.concatenate((best, strided))
    vis_pos = rollouts[vis_idx, :, :3]
    vis_weights = weights[vis_idx]
    # Predicted path of the chosen mean action sequence
    mean_path = rollout(x0, U_new, params, dt)[:, :3]

    return U_new, ess, lam, vis_pos, vis_weights, mean_path


def build_reference(
    waypoints: NDArray, t_total: float, freq: float, n_extend: int = 0
) -> np.ndarray:
    """Create a time-indexed reference from a cubic spline through the waypoints.

    The waypoints are spaced evenly in time over `t_total`. The spline is sampled at `freq`, so
    row `i` of the result is the reference for time `i / freq`. The reference can be extended past
    the last waypoint with a straight continuation at the final velocity.

    Args:
        waypoints: Positions to pass through with shape (N, 3).
        t_total: Duration of the reference in seconds.
        freq: Sampling frequency of the reference in Hz, i.e. the control rate.
        n_extend: Number of rows appended after the spline ends.

    Returns:
        The reference with shape (T + n_extend, 6) and `T = int(freq * t_total)`. The first three
        columns are positions, the last three are velocities.
    """
    # Spline through the waypoints, with the waypoints evenly spaced in time
    spline = CubicSpline(np.linspace(0, t_total, len(waypoints)), waypoints)
    t = np.arange(int(freq * t_total)) / freq  # one sample time per control step
    ref = np.hstack((spline(t), spline.derivative()(t)))  # [position, velocity] per step
    # Continue straight at the final velocity so the horizon never runs out of reference
    steps = np.arange(1, n_extend + 1)[:, None] / freq
    extension = np.hstack((ref[-1, :3] + steps * ref[-1, 3:], np.tile(ref[-1, 3:], (n_extend, 1))))
    return np.vstack((ref, extension))


class AttitudeMPPI(Controller):
    """Sampling-based MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        # Tunables, overridable through `config.controller.mppi` (used by the benchmark harness)
        overrides = dict(config.controller.get("mppi", {}))
        p = {**DEFAULT_PARAMS, **{k: v for k, v in overrides.items() if k in DEFAULT_PARAMS}}
        unknown = set(overrides) - set(DEFAULT_PARAMS) - {"cost_constants", "cost_weights", "mlc"}
        if unknown:
            raise ValueError(f"Unknown MPPI parameters: {sorted(unknown)}")
        self._p = p
        self._cc = {**COST_CONSTANTS, **dict(overrides.get("cost_constants", {}))}
        self._cw = {**COST_WEIGHTS, **dict(overrides.get("cost_weights", {}))}
        mlc = {**MLC_CONSTANTS, **dict(overrides.get("mlc", {}))}

        self._freq = config.env.freq
        self._dt = 1 / self._freq
        self._horizon = int(p["horizon"])  # planning steps, 0.5 s at 50 Hz
        self._n_samples = int(p["n_samples"])  # sampled control sequences per step
        self._sigma = jnp.asarray(p["sigma"], dtype=jnp.float32)  # noise std: r, p, y, thrust
        # fraction of samples that should effectively contribute to the update
        self._target_ess = float(p["target_ess"])
        self._step_ms = []  # wall time of every MPPI update, for benchmarking
        self._lam = float("nan")  # softmin temperature, adapted every step to reach the target ESS
        self._params = load_params(Dynamics.so_rpy_rotor_drag, config.sim.drone)
        # Hover command, inverted from force = acc_coef + cmd_f_coef * cmd
        self._hover_thrust = float(
            (self._params["mass"] * -self._params["gravity_vec"][-1] - self._params["acc_coef"])
            / self._params["cmd_f_coef"]
        )

        self._mlc = MidLevelController(
            SplinePlanner(obs, config),
            v_max=mlc["v_max"],
            a_long_max=mlc["a_long_max"],
            a_lat_max=mlc["a_lat_max"],
            a_brake_max=mlc["a_brake_max"],
            v_gate_max=mlc["v_gate_max"],
            gate_window=mlc["gate_window"],
            gate_plateau=mlc["gate_plateau"],
        )

        # n_samples fixes an array shape, so it must be static for jit
        self._update = jax.jit(mppi_update, static_argnames=("n_samples",))
        self._key = jax.random.key(int(p["rng_seed"]))
        # Warm start: hold hover thrust with level attitude
        self._U_nom = jnp.tile(jnp.array([0.0, 0.0, 0.0, self._hover_thrust]), (self._horizon, 1))
        self._u_prev = self._U_nom[0]
        # The thrust is not observed, so it is estimated by passing the commands through the lag
        self._thrust = 0.0
        self._ess = float("nan")
        self._vis = None  # (rollout positions, rollout weights, mean path), set in compute_control
        self._finished = False
        self._gate_now = None  # (positions, quaternions) of the currently perceived gates
        self._gate_old = []  # per gate, the poses that were perceived before the current one
        self._obstacle_now = None  # positions of the currently perceived obstacles
        self._obstacle_old = []  # per obstacle, the positions perceived before the current one

        self._ref = self._mlc.get_trajectory(obs, self._horizon, self._dt)

        # Warm-up call so the jit compilation happens here and not in the first control step
        jax.block_until_ready(self._run_update(self._key, obs))

    def _run_update(self, key: Array, obs: dict[str, NDArray[np.floating]]) -> tuple:
        """Run one jitted MPPI update for the current observation and nominal sequence."""
        return self._update(
            key,
            self._state(obs),
            self._U_nom,
            self._u_prev,
            self._ref,
            self._get_gates_pos_and_normal(obs),
            jnp.asarray(obs["obstacles_pos"]),
            self._params,
            self._dt,
            self._n_samples,
            self._sigma,
            self._target_ess,
            self._cc,
            self._cw,
        )

    def _state(self, obs: dict[str, NDArray[np.floating]]) -> Array:
        """Convert an observation to the planning state [pos, rpy, vel, drpy, thrust]."""
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        # The model uses roll/pitch/yaw rates, the observation gives the body angular velocity
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        return jnp.asarray(
            np.concatenate((obs["pos"], rpy, obs["vel"], drpy, [self._thrust])), dtype=jnp.float32
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next roll, pitch, yaw and collective thrust command.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The command [roll, pitch, yaw, thrust] as a numpy array.
        """
        if self._mlc._current_progress >= self._mlc._planner._current_reference.length:
            self._finished = True

        self._ref = self._mlc.get_trajectory(obs, self._horizon, self._dt)
        self._track_gate_perception(obs)
        self._track_obstacle_perception(obs)

        # Fresh noise every step: split the key, keep one half and use the other
        self._key, subkey = jax.random.split(self._key)
        t_start = time.perf_counter()
        U_new, ess, lam, vis_pos, vis_weights, mean_path = jax.block_until_ready(
            self._run_update(subkey, obs)
        )
        self._step_ms.append(1e3 * (time.perf_counter() - t_start))
        self._ess = float(ess)
        self._lam = float(lam)
        self._vis = (np.asarray(vis_pos), np.asarray(vis_weights), np.asarray(mean_path))

        self._u_prev = U_new[0]
        # Shift by one step for the next call and repeat the last input to fill the end
        self._U_nom = jnp.concatenate((U_new[1:], U_new[-1:]))
        # Only the first input is applied, the rest is re-planned at the next step
        action = np.asarray(U_new[0], dtype=np.float32)
        # Propagate the thrust estimate with the same first-order lag the model uses
        thrust_dyn_coef = float(self._params["thrust_dyn_coef"])
        self._thrust += thrust_dyn_coef * (action[3] - self._thrust) * self._dt
        return action

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
            True if the controller is finished, False otherwise.
        """
        # `or`, not `|`: `|` binds tighter than `==` and compared `finished | n_passed` to the count
        return self._finished or int(obs["n_gates_passed"]) == len(obs["gate_sequence"])

    def render_callback(self, sim: Sim):
        """Draw the reference by speed, the sampled rollouts by weight and the mean action path.

        Items that no longer fit into the viewer's scene are skipped, the rollouts first.
        """
        if sim.viewer is None:
            # The viewer is created after the first callback and sizes its scene from this value
            sim.max_visual_geom = max(sim.max_visual_geom, VIS_MAX_GEOM)
            return

        # Reference path from high level planner
        ref_path, _ = self._mlc._planner._current_reference.sample_densely(num_samples=200)
        draw_line(sim, ref_path, rgba=(0.0, 1.0, 0.0, 1.0))

        # Local reference trajectory handed to MPPI, colored by its speed
        ref_pos = self._ref[:, :3]
        colors = speed_to_rgba(np.linalg.norm(self._ref[:, 3:6], axis=1)[:-1])
        for i in range(len(ref_pos) - 1):
            draw_line(sim, ref_pos[i : i + 2], rgba=colors[i], start_size=3.0, end_size=3.0)

        self._draw_gate_perception(sim)
        self._draw_obstacle_perception(sim)

        if self._vis is None:
            return
        # Reference point the drone should be tracking right now
        draw_points(sim, ref_pos[:1], rgba=(1.0, 0.9, 0.0, 1.0), size=0.03)

        vis_pos, vis_weights, mean_path = self._vis
        # Rollout tree: opaque, thick and bright for high weight, faint and thin for low weight
        # (sqrt compresses the weights, which span orders of magnitude)
        w = np.sqrt(vis_weights / vis_weights.max())
        reserved = 2 * len(mean_path)  # geoms kept free for the mean path's line and points
        for pos, wi in zip(vis_pos, w):
            if _free_geoms(sim) - reserved < len(pos[::2]) - 1:
                break  # the scene is full, so the remaining rollouts are left out
            draw_line(
                sim,
                pos[::2],
                rgba=(0.1, 1.0, 0.1, 0.1 + 0.9 * wi),
                start_size=1.0 + 3.0 * wi,
                end_size=1.0 + 3.0 * wi,
            )
        # Chosen mean action sequence, drawn last so it is on top
        draw_line(sim, mean_path, rgba=(1.0, 0.0, 1.0, 1.0), start_size=4.5, end_size=4.5)
        draw_points(sim, mean_path, rgba=(1.0, 0.0, 1.0, 1.0), size=0.01)

    def _track_gate_perception(self, obs: dict[str, NDArray[np.floating]]):
        """Store the previously perceived pose of every gate whose observed pose just changed."""
        pos, quat = np.array(obs["gates_pos"]), np.array(obs["gates_quat"])
        if self._gate_now is None:
            self._gate_old = [[] for _ in pos]
        else:
            for i in range(len(pos)):
                same = np.allclose(pos[i], self._gate_now[0][i], atol=1e-6) and np.allclose(
                    quat[i], self._gate_now[1][i], atol=1e-6
                )
                if not same:
                    self._gate_old[i].append((self._gate_now[0][i], self._gate_now[1][i]))
        self._gate_now = (pos, quat)

    def _draw_gate_perception(self, sim: Sim):
        """Draw the current gate perception boldly and earlier perceptions faintly with a link."""
        if self._gate_now is None:
            return
        cyan, orange = (0.0, 0.9, 1.0, 1.0), (1.0, 0.5, 0.0, 0.4)
        tolerance = self._cc["gate_tolerance"]
        for i, (pos, quat) in enumerate(zip(*self._gate_now)):
            # Earlier perceptions, linked to where the gate is perceived now
            for old_pos, old_quat in self._gate_old[i]:
                draw_line(
                    sim,
                    gate_outline(old_pos, old_quat, GATE_OPENING_HALF),
                    rgba=orange,
                    start_size=1.5,
                    end_size=1.5,
                )
                draw_line(sim, np.stack((old_pos, pos)), rgba=orange, start_size=1.5, end_size=1.5)
            # Current perception: opening and the tolerance square the gate cost treats as free
            draw_line(
                sim,
                gate_outline(pos, quat, GATE_OPENING_HALF),
                rgba=cyan,
                start_size=4.0,
                end_size=4.0,
            )
            draw_line(
                sim, gate_outline(pos, quat, tolerance), rgba=cyan, start_size=2.0, end_size=2.0
            )

    def _track_obstacle_perception(self, obs: dict[str, NDArray[np.floating]]):
        """Store the previously perceived position of every obstacle whose observed position changed."""
        pos = np.array(obs["obstacles_pos"])
        if self._obstacle_now is None:
            self._obstacle_old = [[] for _ in pos]
        else:
            for i in range(len(pos)):
                if not np.allclose(pos[i], self._obstacle_now[i], atol=1e-6):
                    self._obstacle_old[i].append(self._obstacle_now[i])
        self._obstacle_now = pos

    def _draw_obstacle_perception(self, sim: Sim):
        """Draw each obstacle's pole and penalty tube, current boldly and earlier ones faintly."""
        if self._obstacle_now is None:
            return
        cyan, orange = (0.0, 0.9, 1.0, 1.0), (1.0, 0.5, 0.0, 0.4)
        radius = self._cc["obstacle_radius"]
        z_tube = float(self._ref[0, 2])  # tube is drawn at the height the drone is flying at
        for i, pos in enumerate(self._obstacle_now):
            # Earlier perceptions, linked to where the obstacle is perceived now
            for old_pos in self._obstacle_old[i]:
                pole = np.array([[*old_pos[:2], 0.0], old_pos])
                draw_line(sim, pole, rgba=orange, start_size=1.5, end_size=1.5)
                draw_line(
                    sim,
                    circle_outline(old_pos[:2], z_tube, radius),
                    rgba=orange,
                    start_size=1.5,
                    end_size=1.5,
                )
                draw_line(
                    sim,
                    np.stack(([*old_pos[:2], z_tube], [*pos[:2], z_tube])),
                    rgba=orange,
                    start_size=1.5,
                    end_size=1.5,
                )
            # Current perception: the pole and the tube radius the obstacle cost penalizes
            draw_line(
                sim, np.array([[*pos[:2], 0.0], pos]), rgba=cyan, start_size=4.0, end_size=4.0
            )
            draw_line(
                sim,
                circle_outline(pos[:2], z_tube, radius),
                rgba=cyan,
                start_size=2.0,
                end_size=2.0,
            )

    def episode_callback(self):
        """Reset the internal state."""
        self._finished = False
        self._gate_now = None
        self._obstacle_now = None
        self._U_nom = jnp.tile(jnp.array([0.0, 0.0, 0.0, self._hover_thrust]), (self._horizon, 1))
        self._u_prev = self._U_nom[0]
        self._thrust = 0.0

    def _get_gates_pos_and_normal(self, obs: dict[str, NDArray[np.floating]]) -> Array:
        """Return the position and the normal vector of every gate with shape (n_gates, 6)."""
        # The gate normal is the gate's local x-axis in the world frame
        gates_normal = R.from_quat(obs["gates_quat"]).apply([1, 0, 0])
        return jnp.hstack([obs["gates_pos"], gates_normal])
