"""Sampling-based MPC (MPPI) with the contouring cost (MPCC) of `mpcc/cost.py`.

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

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.attitude_mppi import adaptive_lambda, rollout, speed_to_rgba
from lsy_drone_racing.control.mpcc.cost import MPCC_WEIGHTS, mpcc_cost, progress_step
from lsy_drone_racing.control.mpcc.path import path_from_reference, path_lookup
from lsy_drone_racing.control.planner.splines import SplinePlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from jax import Array
    from numpy.typing import NDArray

    from lsy_drone_racing.control.mpcc.path import ArcLengthPath

V_THETA_MAX = 6.0  # Maximum progress speed in m/s
DV_THETA_MAX = 10.0  # Maximum magnitude of the progress acceleration in m/s^2
PROGRESS_WINDOW = (0.2, 0.2)  # (behind, ahead) in meters around the predicted progress to search

COLLISION_CONSTANTS = {
    "gate_tolerance": 0.05,  # Acceptable deviation in meters from the gate's center
    "gate_slab": 0.1,  # Half-thickness in meters around a gate's plane in which the gate is checked
    "gate_frame_limit": 0.45,  # Offset in meters from the gate's center beyond the frame
    "obstacle_radius": 0.2,  # Radius in meters of the penalty tube around an obstacle
}

COLLISION_WEIGHTS = {
    "gate_weight": 4.0,  # Weight for gate pass / collision / miss
    "obstacle_weight": 3.0,  # Weight for coming close to an obstacle
}


def collision_cost(pos: Array, gates: Array, obstacles_pos: Array) -> Array:
    """Penalise positions that hit a gate's frame or come close to an obstacle.

    Args:
        pos: Positions with shape (H + 1, 3).
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3). Obstacles are vertical
            poles, so only the x and y coordinates are used.

    Returns:
        The scalar cost.
    """
    # Gates: positions near a gate's plane must be close to its center
    gate_pos = gates[:, :3]
    gate_normal = gates[:, 3:]
    gate_offset = pos[:, None, :] - gate_pos[None, :, :]
    distance = jnp.sum(gate_offset * gate_normal[None, :, :], axis=-1)
    near_gate_mask = jnp.abs(distance) < COLLISION_CONSTANTS["gate_slab"]
    # Max-norm of the offset along the gate plane's vertical and horizontal axes
    vertical_distance = gate_offset[..., 2]
    horizontal_axis = jnp.cross(gate_normal, jnp.array([0, 0, 1]))
    horizontal_distance = jnp.sum(gate_offset * horizontal_axis[None, :, :], axis=-1)
    gate_plane_max_distance = jnp.maximum(jnp.abs(vertical_distance), jnp.abs(horizontal_distance))
    # Positions beside the frame, outside of its outer edge, are free
    in_frame = gate_plane_max_distance < COLLISION_CONSTANTS["gate_frame_limit"]
    frame_mask = near_gate_mask & in_frame
    beyond_tolerance = jnp.maximum(
        gate_plane_max_distance / COLLISION_CONSTANTS["gate_tolerance"] - 1, 0
    )
    gate_cost = jnp.sum(jnp.where(frame_mask, beyond_tolerance**2, 0))

    # Obstacles: tube around each pole, checked in the x-y plane, rising to 1 at the pole's axis
    obstacle_xy_distance = jnp.linalg.norm(pos[:, None, :2] - obstacles_pos[None, :, :2], axis=-1)
    tube_penetration = jnp.maximum(
        1 - obstacle_xy_distance / COLLISION_CONSTANTS["obstacle_radius"], 0
    )
    obstacle_cost = jnp.sum(tube_penetration**2)

    return (
        COLLISION_WEIGHTS["gate_weight"] * gate_cost
        + COLLISION_WEIGHTS["obstacle_weight"] * obstacle_cost
    )


def progress_rollout(
    theta0: Array, v_theta0: Array, dv_theta: Array, length: Array, dt: float
) -> tuple[Array, Array, Array]:
    """Simulate the progress states for a sequence of progress accelerations.

    Args:
        theta0: Initial progress in meters.
        v_theta0: Initial progress speed in m/s.
        dv_theta: Progress accelerations with shape (H,).
        length: Path length in meters.
        dt: Integration step in seconds.

    Returns:
        - Progress with shape (H + 1,), including theta0.
        - Progress speed with shape (H + 1,), including v_theta0.
        - The effective progress accelerations with shape (H,), after the speed limits.
    """

    def step_fn(carry: tuple[Array, Array], dv: Array) -> tuple[tuple, tuple]:
        theta_next, v_theta_next = progress_step(*carry, dv, dt, length, V_THETA_MAX)
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
    gates: Array,
    obstacles_pos: Array,
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
        gates: Position and normal vector of every gate with shape (n_gates, 6).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3).
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
    theta, v_theta, dv_theta = jax.vmap(progress_rollout, in_axes=(None, None, 0, None, None))(
        theta0, v_theta0, sampled_dv_theta, path.length, dt
    )

    def cost_fn(X: Array, U: Array, theta: Array, v_theta: Array, dv_theta: Array) -> Array:
        contouring = mpcc_cost(X, U, u_prev, theta, v_theta, dv_theta, path, gates[:, :3], weights)
        return contouring + collision_cost(X[:, :3], gates, obstacles_pos)

    costs = jax.vmap(cost_fn)(rollouts, sampled_actions, theta, v_theta, dv_theta)

    lam = adaptive_lambda(costs, target_ess)
    sample_weights = jax.nn.softmax(-costs / lam)
    ess = 1 / jnp.sum(sample_weights**2)

    U_new = jnp.sum(sample_weights[:, None, None] * sampled_actions, axis=0)
    # Average the effective accelerations, so the plan matches what the progress states did
    dv_theta_new = jnp.sum(sample_weights[:, None] * dv_theta, axis=0)
    theta_plan, v_theta_plan, _ = progress_rollout(theta0, v_theta0, dv_theta_new, path.length, dt)

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


class AttitudeMPPIMPCC(Controller):
    """MPPI on the contouring cost using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._dt = 1 / config.env.freq
        self._horizon = 30  # planning steps, 0.6 s at 50 Hz
        self._n_samples = 1024  # sampled sequences per step
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

        self._planner = SplinePlanner(obs, config)
        self._reference = self._planner.plan(obs)
        self._path = path_from_reference(self._reference)
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
            jnp.asarray(np.hstack((obs["gates_pos"], gates_normal))),
            jnp.asarray(obs["obstacles_pos"]),
            MPCC_WEIGHTS,
            self._params,
            self._dt,
            self._n_samples,
            self._sigma,
            self._sigma_dv_theta,
            self._target_ess,
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
        # The planner updates the reference in place when gates or obstacles are revealed
        self._reference = self._planner.plan(obs)
        if self._reference._version != self._path_version:
            self._path = path_from_reference(self._reference)
            self._path_version = self._reference._version

        # Measure the progress near the one predicted by the last plan
        theta0 = self._reference.get_progress(
            obs["pos"], prev_progress=self._theta_pred, search_range_s=PROGRESS_WINDOW
        )
        if theta0 >= self._reference.length:
            self._finished = True

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

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Check whether the path's end is reached or all gates are passed.

        Returns:
            True if the controller is finished, False otherwise.
        """
        return self._finished or int(obs["n_gates_passed"]) == len(obs["gate_sequence"])

    def render_callback(self, sim: Sim):
        """Draw the planned path by progress speed, the sampled rollouts and the mean path."""
        if self._vis is None:
            return
        plan_path, v_theta_plan, vis_pos, vis_weights, mean_path = self._vis

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
