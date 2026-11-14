"""Sampling-based MPC (MPPI) with the contouring cost (MPCC) of `mpcc/cost.py`.

The path to follow comes from any planner implementing `planner/interface.py`, so this module
does not depend on how the path is generated.

Instead of tracking a time-indexed reference, the controller samples a virtual progress
acceleration dv_theta next to the commands, so it chooses itself how fast to move along the planned
path. The progress theta of the first state is measured by projecting the drone onto the path every
step, so replans that change the path's arc length need no extra handling.

State and input conventions follow `attitude_mppi.py`:

    x = [pos(3), rpy(3), vel(3), drpy(3)]
    u = [roll, pitch, yaw, thrust]  (thrust is the total collective thrust in N)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.attitude_mppi import adaptive_lambda, rollout, speed_to_rgba
from lsy_drone_racing.control.mpcc.cost import MPCC_WEIGHTS, mpcc_cost, progress_step
from lsy_drone_racing.control.mpcc.path import (
    carry_progress,
    path_from_reference,
    path_lookup,
    profile_lookup,
    speed_profile,
)
from lsy_drone_racing.control.utils.collision import (
    gate_contact,
    obstacle_clearance,
    obstacle_position_uncertainty,
)
from lsy_drone_racing.control.utils.utils import drone_collision_radius, obstacle_radius

if TYPE_CHECKING:
    from crazyflow import Sim
    from jax import Array
    from numpy.typing import NDArray

    from lsy_drone_racing.control.mpcc.path import ArcLengthPath
    from lsy_drone_racing.control.planner.interface import PathPlanner

V_THETA_MAX = 15.0  # Maximum progress speed in m/s
# Progress speed cap where the path comes within SLOW_RADIUS (horizontally, in meters) of a gate
# whose true pose is not revealed yet. The pose is revealed at 0.7 m, and arriving slowly leaves
# time to correct towards it.
V_THETA_SLOW = 2
SLOW_RADIUS = 1.0
DV_THETA_MAX = 12.0  # Maximum magnitude of the progress acceleration in m/s^2
# Speed profile of the progress, see `mpcc/path.speed_profile`: sideways acceleration in m/s^2 in
# turns, v <= sqrt(A_LAT_MAX / k) for the path's curvature k, and deceleration in m/s^2 before them.
# Both are below the ~15.6 m/s^2 the drone can produce sideways while holding its altitude, leaving
# room for tracking errors.
A_LAT_MAX = 12.0
A_BRAKE = 6.0
CURVATURE_DS = 0.05  # Arc length in meters over which the curvature is measured
PROGRESS_WINDOW = (0.2, 0.2)  # (behind, ahead) in meters around the predicted progress to search
# How much faster in m/s than the drone's own speed along the path the progress may start a step.
# The progress speed is a state that only changes by DV_THETA_MAX, so without this bound it can run
# ahead of the drone and leave every sampled rollout with a reference far ahead of it.
V_THETA_SLACK = 1.0

# Obstacle and crash cost, after Williams et al., "Model Predictive Path Integral Control using
# Covariance Variable Importance Sampling", 2015: w_obs * exp(-d^2 / (2 sigma^2)) on the clearance d
# to the nearest obstacle, and w_crash for every step after a crash.
COLLISION_WEIGHTS = {
    # Obstacle cost at zero clearance. Kept small: the planned path keeps only about 0.1 m of
    # clearance from obstacles, and a strong cost there makes hovering cheaper than progressing.
    "w_obs": 30.0,
    "obstacle_sigma": 0.1,  # Decay length in meters of the obstacle cost
    "w_crash": 3000.0,  # Cost per step from the first crash to the end of the horizon
}


def crash_mask(
    pos: Array, clearance: Array, gates: Array, pos_limits: Array, drone_radius: float
) -> Array:
    """Flag every step from the first crash on.

    A crash is contact with an obstacle or a gate, or leaving the position limits, which also
    covers hitting the floor. The initial state is never flagged, so all rollouts are not equally
    crashed when the drone starts on the ground or the contact model is conservative.

    Args:
        pos: Positions with shape (H + 1, 3).
        clearance: Clearance to the nearest obstacle with shape (H + 1,), see `obstacle_clearance`.
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        pos_limits: Lower and upper position limits with shape (2, 3).
        drone_radius: Collision radius of the drone in meters.

    Returns:
        Crash flags with shape (H + 1,), True from the first crash to the end.
    """
    outside = jnp.any((pos < pos_limits[0]) | (pos > pos_limits[1]), axis=-1)
    contact = (clearance <= 0) | gate_contact(pos, gates, drone_radius) | outside
    contact = contact.at[0].set(False)
    return jnp.cumsum(contact) > 0


def freeze_after_crash(X: Array, crashed: Array) -> Array:
    """Hold the state of the first crashed step for the rest of the rollout.

    Args:
        X: State trajectory with shape (H + 1, 12).
        crashed: Crash flags with shape (H + 1,), see `crash_mask`.

    Returns:
        The state trajectory with shape (H + 1, 12).
    """
    first_crash = jnp.argmax(crashed)
    return jnp.where(crashed[:, None], X[first_crash], X)


def collision_cost(clearance: Array, crashed: Array, weights: dict) -> Array:
    """Exponential cost on the obstacle clearance plus a sparse cost on crashed steps.

    Args:
        clearance: Clearance to the nearest obstacle with shape (H + 1,).
        crashed: Crash flags with shape (H + 1,).
        weights: Cost weights, see `COLLISION_WEIGHTS`.

    Returns:
        The scalar cost.
    """
    # Clipped so the cost does not fall again inside an obstacle
    d = jnp.maximum(clearance, 0.0)
    obstacle_cost = jnp.sum(jnp.exp(-0.5 * d**2 / weights["obstacle_sigma"] ** 2))
    return weights["w_obs"] * obstacle_cost + weights["w_crash"] * jnp.sum(crashed)


def progress_rollout(
    theta0: Array,
    v_theta0: Array,
    dv_theta: Array,
    path: ArcLengthPath,
    speed_limit: Array,
    gates: Array,
    gates_revealed: Array,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Simulate the progress states for a sequence of progress accelerations.

    The progress speed is capped at V_THETA_SLOW where the path is near an unrevealed gate, and at
    V_THETA_MAX elsewhere, and everywhere by the path's speed profile.

    Args:
        theta0: Initial progress in meters.
        v_theta0: Initial progress speed in m/s.
        dv_theta: Progress accelerations with shape (H,).
        path: The path to follow.
        speed_limit: Speed profile of the path in m/s on its grid, see `speed_profile`.
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        gates_revealed: Whether each gate's true pose is known, with shape (n_gates,).
        dt: Integration step in seconds.

    Returns:
        - Progress with shape (H + 1,), including theta0.
        - Progress speed with shape (H + 1,), including v_theta0.
        - The effective progress accelerations with shape (H,), after the speed limits.
    """

    def step_fn(carry: tuple[Array, Array], dv: Array) -> tuple[tuple, tuple]:
        path_xy = path_lookup(path, carry[0])[0][:2]
        gate_xy_distance = jnp.linalg.norm(gates[:, :2] - path_xy, axis=-1)
        near_unrevealed = jnp.any((gate_xy_distance < SLOW_RADIUS) & ~gates_revealed)
        v_theta_max = jnp.where(near_unrevealed, V_THETA_SLOW, V_THETA_MAX)
        v_theta_max = jnp.minimum(v_theta_max, profile_lookup(speed_limit, path.length, carry[0]))
        theta_next, v_theta_next = progress_step(*carry, dv, dt, path.length, v_theta_max)
        return (theta_next, v_theta_next), (theta_next, v_theta_next)

    _, (theta, v_theta) = jax.lax.scan(step_fn, (theta0, v_theta0), dv_theta)
    theta = jnp.concatenate((theta0[None], theta))
    v_theta = jnp.concatenate((v_theta0[None], v_theta))
    return theta, v_theta, jnp.diff(v_theta) / dt


N_VIS_BEST = 10  # lowest-cost rollouts that are drawn
N_VIS_RANDOM = 30  # additional evenly strided rollouts that are drawn


def mppi_mpcc_update(
    key: Array,
    x0: Array,
    theta0: Array,
    v_theta0: Array,
    U_nom: Array,
    dv_theta_nom: Array,
    u_prev: Array,
    path: ArcLengthPath,
    speed_limit: Array,
    gates: Array,
    gates_revealed: Array,
    obstacles_pos: Array,
    obstacles_inflation: Array,
    pos_limits: Array,
    drone_radius: Array,
    obstacle_radius: Array,
    weights: dict,
    params: dict,
    dt: float,
    n_samples: int,
    sigma: Array,
    sigma_dv_theta: float,
    target_ess: float,
) -> tuple[Array, ...]:
    """Perform one MPPI iteration on the MPCC cost.

    Samples `n_samples` perturbed command and progress acceleration sequences around the nominal
    ones, simulates the drone and the progress states, and returns the nominal sequences updated
    with the softmin-weighted samples.

    Args:
        key: JAX PRNG key.
        x0: Current state with shape (12,).
        theta0: Current progress in meters.
        v_theta0: Current progress speed in m/s.
        U_nom: Nominal command sequence with shape (H, 4).
        dv_theta_nom: Nominal progress acceleration sequence with shape (H,).
        u_prev: Previous command with shape (4,).
        path: The path to follow.
        speed_limit: Speed profile of the path in m/s on its grid, see `speed_profile`.
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        gates_revealed: Whether each gate's true pose is known, with shape (n_gates,).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3).
        obstacles_inflation: Extra obstacle radius in meters with shape (n_obstacles,), see
            `obstacle_clearance`.
        pos_limits: Lower and upper position limits with shape (2, 3).
        drone_radius: Collision radius of the drone in meters.
        obstacle_radius: Collision radius of the obstacle poles in meters.
        weights: MPCC cost weights, see `MPCC_WEIGHTS`.
        params: Drone parameters.
        dt: Integration step in seconds.
        n_samples: Number of sampled sequences.
        sigma: Per-channel standard deviation of the command noise with shape (4,).
        sigma_dv_theta: Standard deviation of the progress acceleration noise.
        target_ess: Desired effective sample size as a fraction of n_samples.

    Returns:
        - U_new: The updated command sequence with shape (H, 4).
        - dv_theta_new: The updated progress acceleration sequence with shape (H,).
        - theta_plan: Progress of the updated plan with shape (H + 1,).
        - v_theta_plan: Progress speed of the updated plan with shape (H + 1,).
        - plan_path: Path positions at the plan's progress with shape (H + 1, 3).
        - ess: Effective sample size for debugging and logging.
        - lam: The chosen softmin temperature for debugging and logging.
        - vis_pos: Positions of the visualized rollouts with shape (N_VIS_BEST + N_VIS_RANDOM,
          H + 1, 3).
        - vis_weights: Softmin weights of the visualized rollouts.
        - mean_path: Positions of the rollout of `U_new` with shape (H + 1, 3).
    """
    horizon = len(U_nom)
    key_u, key_dv = jax.random.split(key)
    sampled_actions = U_nom + sigma * jax.random.normal(key_u, (n_samples, horizon, 4))
    sampled_actions = jnp.clip(
        sampled_actions,
        jnp.array([-1.0, -1.0, -1.0, 4 * params["thrust_min"]]),
        jnp.array([1.0, 1.0, 1.0, 4 * params["thrust_max"]]),
    )
    sampled_dv_theta = dv_theta_nom + sigma_dv_theta * jax.random.normal(
        key_dv, (n_samples, horizon)
    )
    sampled_dv_theta = jnp.clip(sampled_dv_theta, -DV_THETA_MAX, DV_THETA_MAX)

    rollouts = jax.vmap(rollout, in_axes=(None, 0, None, None))(x0, sampled_actions, params, dt)
    theta, v_theta, dv_theta = jax.vmap(
        progress_rollout, in_axes=(None, None, 0, None, None, None, None, None)
    )(theta0, v_theta0, sampled_dv_theta, path, speed_limit, gates, gates_revealed, dt)

    def cost_fn(X: Array, U: Array, theta: Array, v_theta: Array, dv_theta: Array) -> Array:
        radius = drone_radius + obstacle_radius
        clearance = obstacle_clearance(X[:, :3], obstacles_pos, obstacles_inflation, radius)
        crashed = crash_mask(X[:, :3], clearance, gates, pos_limits, drone_radius)
        # A crashed rollout stays where it crashed and earns no progress from then on
        X = freeze_after_crash(X, crashed)
        clearance = obstacle_clearance(X[:, :3], obstacles_pos, obstacles_inflation, radius)
        contouring = mpcc_cost(
            X, U, u_prev, theta, v_theta * ~crashed, dv_theta, path, gates[:, :3], weights
        )
        return contouring + collision_cost(clearance, crashed, COLLISION_WEIGHTS)

    costs = jax.vmap(cost_fn)(rollouts, sampled_actions, theta, v_theta, dv_theta)

    lam = adaptive_lambda(costs, target_ess)
    sample_weights = jax.nn.softmax(-costs / lam)
    ess = 1 / jnp.sum(sample_weights**2)

    U_new = jnp.sum(sample_weights[:, None, None] * sampled_actions, axis=0)
    # Average the effective accelerations, so the plan matches what the progress states did
    dv_theta_new = jnp.sum(sample_weights[:, None] * dv_theta, axis=0)
    theta_plan, v_theta_plan, _ = progress_rollout(
        theta0, v_theta0, dv_theta_new, path, speed_limit, gates, gates_revealed, dt
    )

    best = jax.lax.top_k(-costs, N_VIS_BEST)[1]
    strided = jnp.arange(N_VIS_RANDOM) * (n_samples // N_VIS_RANDOM)
    vis_idx = jnp.concatenate((best, strided))
    mean_path = rollout(x0, U_new, params, dt)[:, :3]

    return (
        U_new,
        dv_theta_new,
        theta_plan,
        v_theta_plan,
        path_lookup(path, theta_plan)[0],
        ess,
        lam,
        rollouts[vis_idx, :, :3],
        sample_weights[vis_idx],
        mean_path,
    )


def path_speed_profile(path: ArcLengthPath) -> Array:
    """Speed profile of a path with this module's limits, see `mpcc/path.speed_profile`."""
    return speed_profile(path, A_LAT_MAX, A_BRAKE, V_THETA_MAX, CURVATURE_DS)


class MPPIMPCC:
    """MPPI on the contouring cost using the collective thrust and attitude interface.

    Holds the optimizer state between control steps. Controllers wrap it with their planner.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], config: dict, planner: PathPlanner):
        """Initialize the optimizer and compile the update.

        Args:
            obs: The initial observation of the environment's state.
            config: The race configuration.
            planner: Planner of the path to follow.
        """
        self._dt = 1 / config.env.freq
        self._horizon = 25  # planning steps, 0.5 s at 50 Hz
        self._n_samples = 2048  # sampled sequences per step
        # Noise std per channel: roll, pitch, yaw, thrust
        self._sigma = jnp.array([0.15, 0.15, 0.0, 0.05]) * 2.0
        self._sigma_dv_theta = 2.0  # noise std of the progress acceleration in m/s^2
        self._target_ess = 0.05  # fraction of samples that should effectively contribute
        self._params = load_params(Dynamics.so_rpy, config.sim.drone)
        # Hover command, inverted from force = acc_coef + cmd_f_coef * cmd
        self._hover_thrust = float(
            (self._params["mass"] * -self._params["gravity_vec"][-1] - self._params["acc_coef"])
            / self._params["cmd_f_coef"]
        )

        limits = config.env.track.safety_limits
        self._pos_limits = jnp.array([limits.pos_limit_low, limits.pos_limit_high])
        self._drone_radius = jnp.float32(drone_collision_radius(config.sim.drone))
        self._obstacle_radius = jnp.float32(obstacle_radius())
        self._obstacle_uncertainty = obstacle_position_uncertainty(config)

        self._planner = planner
        self._reference = self._planner.plan(obs)
        self._path = path_from_reference(self._reference)
        self._speed_limit = path_speed_profile(self._path)
        self._path_version = self._reference._version

        self._update = jax.jit(mppi_mpcc_update, static_argnames=("n_samples",))
        self._key = jax.random.key(0)
        # Warm start: hold hover thrust with level attitude, at zero progress speed
        self._U_nom = jnp.tile(jnp.array([0.0, 0.0, 0.0, self._hover_thrust]), (self._horizon, 1))
        self._dv_theta_nom = jnp.zeros(self._horizon)
        self._u_prev = self._U_nom[0]
        self._theta_pred = self._reference.get_progress(obs["pos"])
        self._v_theta_pred = 0.0
        self._finished = False
        self._vis = None  # (plan path, plan speed, rollout positions, weights, mean path)

        # Warm-up call so the jit compilation happens here and not in the first control step
        self._run_update(self._key, obs, self._theta_pred)

    def _state(self, obs: dict[str, NDArray[np.floating]]) -> Array:
        """Convert an observation to the planning state [pos, rpy, vel, drpy]."""
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        # The model uses roll/pitch/yaw rates, the observation gives the body angular velocity
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        return jnp.asarray(np.concatenate((obs["pos"], rpy, obs["vel"], drpy)))

    def _run_update(self, key: Array, obs: dict[str, NDArray[np.floating]], theta0: float) -> tuple:
        """Run one MPPI iteration from the current observation."""
        # The gate normal is the gate's local x-axis in the world frame
        gates_normal = R.from_quat(obs["gates_quat"]).apply([1, 0, 0])
        return self._update(
            key,
            self._state(obs),
            jnp.float32(theta0),
            jnp.float32(self._v_theta_pred),
            self._U_nom,
            self._dv_theta_nom,
            self._u_prev,
            self._path,
            self._speed_limit,
            jnp.asarray(np.hstack((obs["gates_pos"], gates_normal))),
            jnp.asarray(obs["gates_visited"]),
            jnp.asarray(obs["obstacles_pos"]),
            jnp.asarray(
                np.where(obs["obstacles_visited"], 0.0, self._obstacle_uncertainty),
                dtype=jnp.float32,
            ),
            self._pos_limits,
            self._drone_radius,
            self._obstacle_radius,
            MPCC_WEIGHTS,
            self._params,
            self._dt,
            self._n_samples,
            self._sigma,
            self._sigma_dv_theta,
            self._target_ess,
        )

    def _replace_path(self, obs: dict[str, NDArray[np.floating]]):
        """Switch to the planner's new path and move the progress states onto it.

        The command sequence is kept as warm start, since it is still physically sensible.
        """
        new_path = path_from_reference(self._reference)
        if self._planner.starts_at_drone:
            # The path starts at the drone, moving along its first tangent
            self._theta_pred = 0.0
            speed_along_path = float(np.dot(obs["vel"], np.asarray(new_path.tangents[0])))
            self._v_theta_pred = float(np.clip(speed_along_path, 0.0, V_THETA_MAX))
        else:
            self._theta_pred = carry_progress(self._path, self._theta_pred, self._reference)
        self._path = new_path
        self._speed_limit = path_speed_profile(new_path)
        self._path_version = self._reference._version

    def compute_control(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Compute the next roll, pitch, yaw and collective thrust command.

        Args:
            obs: The current observation of the environment.

        Returns:
            The command [roll, pitch, yaw, thrust] as a numpy array.
        """
        # The planner may update the reference in place, so the old path is taken from self._path
        self._reference = self._planner.plan(obs)
        if self._reference._version != self._path_version:
            self._replace_path(obs)

        # Measure the progress near the one predicted by the last plan
        theta0 = self._reference.get_progress(
            obs["pos"], prev_progress=self._theta_pred, search_range_s=PROGRESS_WINDOW
        )
        if theta0 >= self._reference.length:
            self._finished = True
        tangent = np.asarray(path_lookup(self._path, jnp.float32(theta0))[1])
        speed_along_path = max(float(np.dot(obs["vel"], tangent)), 0.0)
        self._v_theta_pred = min(self._v_theta_pred, speed_along_path + V_THETA_SLACK)

        self._key, subkey = jax.random.split(self._key)
        (
            U_new,
            dv_theta_new,
            theta_plan,
            v_theta_plan,
            plan_path,
            _,
            _,
            vis_pos,
            vis_weights,
            mean_path,
        ) = self._run_update(subkey, obs, theta0)
        self._vis = tuple(
            np.asarray(a) for a in (plan_path, v_theta_plan, vis_pos, vis_weights, mean_path)
        )

        self._u_prev = U_new[0]
        self._theta_pred = float(theta_plan[1])
        self._v_theta_pred = float(v_theta_plan[1])
        # Shift by one step for the next call and repeat the last input to fill the end
        self._U_nom = jnp.concatenate((U_new[1:], U_new[-1:]))
        self._dv_theta_nom = jnp.concatenate((dv_theta_new[1:], dv_theta_new[-1:]))
        # Only the first input is applied, the rest is re-planned at the next step
        return np.asarray(U_new[0], dtype=np.float32)

    def is_finished(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Check whether the path's end is reached or all gates are passed.

        Args:
            obs: The current observation of the environment.

        Returns:
            True if the controller is finished, False otherwise.
        """
        return self._finished or int(obs["n_gates_passed"]) == len(obs["gate_sequence"])

    def render(self, sim: Sim):
        """Draw the whole path, the planned part by speed, the rollouts and the mean path."""
        if self._vis is None:
            return
        plan_path, v_theta_plan, vis_pos, vis_weights, mean_path = self._vis

        # Whole path through the remaining gates, thin and faint, drawn first so it is underneath.
        # Coarsely sampled: every segment uses one of the viewer's 1000 geoms, and the rollouts
        # already use most of them.
        whole_path = np.asarray(self._path.positions)[::40]
        draw_line(sim, whole_path, rgba=(1.0, 1.0, 1.0, 0.4), start_size=1.5, end_size=1.5)

        # Part of the path covered by the plan, colored by the planned progress speed
        colors = speed_to_rgba(v_theta_plan[:-1])
        for i in range(len(plan_path) - 1):
            draw_line(sim, plan_path[i : i + 2], rgba=colors[i], start_size=3.0, end_size=3.0)
        # Path point at the current progress
        draw_points(sim, plan_path[:1], rgba=(1.0, 0.9, 0.0, 1.0), size=0.03)

        # Rollout tree: opaque, thick and bright for high weight, faint and thin for low weight
        # (sqrt compresses the weights, which span orders of magnitude)
        w = np.sqrt(vis_weights / vis_weights.max())
        for pos, wi in zip(vis_pos, w):
            draw_line(
                sim,
                pos[::2],
                rgba=(0.1, 1.0, 0.1, 0.1 + 0.9 * wi),
                start_size=1.0 + 3.0 * wi,
                end_size=1.0 + 3.0 * wi,
            )
        # Chosen mean command sequence, drawn last so it is on top
        draw_line(sim, mean_path, rgba=(1.0, 0.0, 1.0, 1.0), start_size=4.5, end_size=4.5)
        draw_points(sim, mean_path, rgba=(1.0, 0.0, 1.0, 1.0), size=0.01)
