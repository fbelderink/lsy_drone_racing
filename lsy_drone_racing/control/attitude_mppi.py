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
from typing import TYPE_CHECKING, Any, NamedTuple

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


def state_derivative(x: Array, u: Array, params: dict) -> Array:
    """Continuous-time derivative of the so_rpy_rotor_drag model.

    Args:
        x: State [pos, rpy, vel, drpy, thrust] with shape (13,).
        u: Input [roll, pitch, yaw, thrust] with shape (4,).
        params: Drone parameters as returned by `crazyflow.dynamics.load_params`.

    Returns:
        The state derivative with shape (13,).
    """
    pos, rpy, vel, drpy, thrust = jnp.split(x, [3, 6, 9, 12])
    return jnp.concatenate(
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


def discretize(params: dict, dts: NDArray) -> dict:
    """Exact zero-order-hold discretization of the linear attitude and thrust dynamics.

    Every attitude axis is the linear system `[a, a_dot]' = [[0, 1], [k_a, k_r]] [a, a_dot] + [0, k_c] u`
    and the thrust a first-order lag with rate `k_t`. Both are discretized exactly for each step
    length with a matrix exponential, which is stable and exact for any step length (Euler diverges
    for the roll/pitch dynamics near 0.12 s and is poor already at 0.06 s).

    Args:
        params: Drone parameters.
        dts: Step lengths with shape (H,).

    Returns:
        Dict with `Ad` (H, 3, 2, 2), `Bd` (H, 3, 2) and `thrust_gain` (H,), the fraction of the gap
        between command and thrust that is closed within a step.
    """
    from scipy.linalg import expm

    k_a, k_r, k_c = (np.asarray(params[k]) for k in ("rpy_coef", "rpy_rates_coef", "cmd_rpy_coef"))
    Ad = np.zeros((len(dts), 3, 2, 2))
    Bd = np.zeros((len(dts), 3, 2))
    for i, dt in enumerate(dts):
        for axis in range(3):
            M = np.zeros((3, 3))
            M[0, 1] = 1.0
            M[1] = [k_a[axis], k_r[axis], k_c[axis]]
            E = expm(M * dt)
            Ad[i, axis], Bd[i, axis] = E[:2, :2], E[:2, 2]
    thrust_gain = 1 - np.exp(-float(params["thrust_dyn_coef"]) * np.asarray(dts))
    return {
        "Ad": jnp.asarray(Ad, dtype=jnp.float32),
        "Bd": jnp.asarray(Bd, dtype=jnp.float32),
        "thrust_gain": jnp.asarray(thrust_gain, dtype=jnp.float32),
    }


def dynamics_step(
    x: Array,
    u: Array,
    params: dict,
    dt: Array,
    disc: dict,
    integrator: str = "euler",
    dist_acc: Array | None = None,
) -> Array:
    """Advance the so_rpy_rotor_drag model by one time step.

    Integrators:
        euler: forward Euler on the full state (the original scheme).
        exact: exact discretization of attitude and thrust (`disc`), translation with the
            trapezoidal rule on the accelerations at the start and the end of the step. The
            acceleration varies with attitude and thrust within the step, so this is second order.
        rk4: classic Runge-Kutta on the full state, as an accuracy reference.

    Args:
        x: State [pos, rpy, vel, drpy, thrust] with shape (13,).
        u: Input [roll, pitch, yaw, thrust] with shape (4,).
        params: Drone parameters as returned by `crazyflow.dynamics.load_params`.
        dt: Step length in seconds.
        disc: This step's slice of `discretize` (`Ad` (3, 2, 2), `Bd` (3, 2), `thrust_gain`).
        integrator: One of "euler", "exact", "rk4".
        dist_acc: Constant disturbance acceleration in the world frame with shape (3,), or None.

    Returns:
        The next state with shape (13,).
    """

    def f(x: Array) -> Array:
        x_dot = state_derivative(x, u, params)
        if dist_acc is not None:
            x_dot = x_dot.at[6:9].add(dist_acc)
        return x_dot

    if integrator == "euler":
        return x + dt * f(x)
    if integrator == "rk4":
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if integrator != "exact":
        raise ValueError(f"Unknown integrator {integrator}")
    pos, rpy, vel, drpy, thrust = jnp.split(x, [3, 6, 9, 12])
    att = jnp.stack((rpy, drpy), axis=-1)  # (3, 2): angle and rate per axis
    att_next = jnp.einsum("aij,aj->ai", disc["Ad"], att) + disc["Bd"] * u[:3, None]
    thrust_next = thrust + disc["thrust_gain"] * (u[3] - thrust)
    acc0 = f(x)[6:9]
    # acceleration at the end of the step, with the velocity predicted by Euler
    x_end = jnp.concatenate((pos, att_next[:, 0], vel + dt * acc0, att_next[:, 1], thrust_next))
    acc1 = f(x_end)[6:9]
    vel_next = vel + 0.5 * dt * (acc0 + acc1)
    # exact for an acceleration that changes linearly over the step
    pos_next = pos + dt * vel + dt**2 / 6 * (2 * acc0 + acc1)
    return jnp.concatenate((pos_next, att_next[:, 0], vel_next, att_next[:, 1], thrust_next))


def rollout(
    x0: Array,
    U: Array,
    params: dict,
    dts: Array,
    disc: dict,
    integrator: str = "euler",
    dist_acc: Array | None = None,
) -> Array:
    """Simulate a control sequence starting from x0.

    Args:
        x0: Initial state with shape (13,).
        U: Control sequence with shape (H, 4). Input `k` is held for `dts[k]` seconds.
        params: Drone parameters.
        dts: Step lengths with shape (H,).
        disc: Discretization of every step, see `discretize`.
        integrator: One of "euler", "exact", "rk4", see `dynamics_step`.
        dist_acc: Constant disturbance acceleration with shape (3,), or None.

    Returns:
        The state trajectory with shape (H + 1, 13), including x0.
    """

    def step_fn(x: Array, inputs: tuple) -> tuple[Array, Array]:
        u, dt, d = inputs
        x_next = dynamics_step(x, u, params, dt, d, integrator, dist_acc)
        return x_next, x_next

    _, outputs = jax.lax.scan(step_fn, x0, (U, dts, disc))
    return jnp.vstack((x0, outputs))


def time_grid(dt: float, n_fine: int, n_coarse: int = 0, coarse_factor: int = 3) -> NDArray:
    """Step lengths of a horizon with `n_fine` steps of `dt` followed by `n_coarse` longer steps.

    The coarse steps are an integer multiple of `dt`, so every knot falls onto the control rate and
    the time-indexed reference can be sampled without interpolation.

    Returns:
        The step lengths with shape (n_fine + n_coarse,).
    """
    return np.array([dt] * n_fine + [coarse_factor * dt] * n_coarse, dtype=np.float32)


def shift_controls(U: Array, dts: Array, shift: float) -> Array:
    """Shift a control sequence on a (possibly non-uniform) grid forward in time by `shift`.

    Input `k` starts at `t_k = sum(dts[:k])`. The shifted sequence holds the input of the original
    piecewise-constant sequence at `t_k + shift`, i.e. at the start of each new step. On a uniform
    grid with `shift = dt` this drops the first input and repeats the last one.
    """
    t = jnp.concatenate((jnp.zeros(1), jnp.cumsum(dts)[:-1]))
    # Tolerance of a quarter of the shortest step against rounding in the cumulative sum
    tol = 0.25 * jnp.min(dts)
    idx = jnp.clip(jnp.searchsorted(t, t + shift + tol, side="right") - 1, 0, len(dts) - 1)
    return U[idx]


COST_CONSTANTS = {
    "deviation_tolerance": 0.13,  # Tolerance in meters for an acceptable deviation from the reference trajectory
    "velocity_tolerance": 0.5,  # Tolerance in m/s for an acceptable deviation from the reference velocity
    "smoothness_tolerance": 1.0,  # Tolerance for control smoothness in multiples of sigma
    "gate_tolerance": 0.07,  # Tolerance in meters for an acceptable deviation from the gate's center
    "gate_slab": 0.1,  # Half-thickness in meters around a gate's plane in which the gate is checked
    "gate_frame_limit": 0.45,  # Offset in meters from the gate's center beyond which the frame is cleared
    "obstacle_radius": 0.2,  # Radius in meters of the penalty tube around an obstacle
    "gate_crossing_scale": 4.0,  # Weight of one plane crossing relative to one state in the slab
    "contour_tolerance": 0.1,  # Tolerance in meters for the distance to the path (contouring cost)
    "gate_bonus_radius": 0.12,  # Max. offset in meters from the gate's center for the gate bonus
}

COST_WEIGHTS = {
    "deviation_weight": 0.2,  # Weight for deviation from reference trajectory
    "velocity_weight": 0.1,  # Weight for deviation from reference velocity
    "smoothness_weight": 0.01,  # Weight for control smoothness
    "gate_weight": 100.0,  # Weight for gate pass / collision / miss
    "obstacle_weight": 50.0,  # Weight for coming close to an obstacle
    # Contouring cost, used if `cost_mode` is "contouring" or "hybrid"
    "contour_weight": 0.2,  # Weight for the distance to the path
    "progress_weight": 10.0,  # Reward per meter of progress at the end of the horizon
    "speed_cap_weight": 1.0,  # Weight for exceeding the speed profile
    "gate_bonus_weight": 20.0,  # Reward for crossing the next gate near its center
}

DEFAULT_PARAMS = {
    "horizon": 30,  # planning steps, 0.6 s at 50 Hz
    "n_samples": 1024,  # sampled control sequences per step
    "sigma": [0.9, 0.9, 0.0, 0.3],  # noise std: roll, pitch, yaw (rad), thrust (N)
    "target_ess": 0.05,  # fraction of samples that should effectively contribute to the update
    "rng_seed": 0,  # seed of the sampling noise
    # Sampling (roadmap item 1)
    "noise_beta": 0.0,  # AR(1) coefficient of temporally smoothed noise, 0 for white noise
    "truncated": False,  # truncated-normal sampling within the action limits instead of clipping
    "include_nominal": False,  # always include the unperturbed nominal and a hover sequence
    "nan_guard": False,  # replace NaN or inf rollout costs with a large constant
    # Horizon (roadmap item 2): `horizon` steps of 1 / freq, then `n_coarse` steps of
    # `coarse_factor` / freq
    "n_coarse": 0,
    "coarse_factor": 3,
    "integrator": "euler",  # "euler", "exact" or "rk4", see `dynamics_step`
    "gate_mode": "slab",  # "slab", "crossing" or "both", see `trajectory_cost`
    "step_weighting": "dt",  # "dt": weight running costs by step length, "none": every knot alike
    # Time constant in seconds of an exponential discount of the running costs, 0 for none. The
    # spread of the rollouts grows with time, so undiscounted far-future costs dominate the cost
    # differences between samples and drown the near-term terms that decide the next gate
    "discount_tau": 0.0,
    # Disturbance estimate (roadmap item 3): low-pass filter of the gap between the observed and
    # the predicted velocity change, fed to the rollouts as a constant acceleration
    "dist_alpha": 0.0,  # filter gain per control step in (0, 1], 0 disables the estimate
    "dist_max": 3.0,  # limit of the estimate per axis in m/s^2
    "dist_axes": [1.0, 1.0, 1.0],  # mask of the estimated axes (x, y, z)
    # Progress-based cost (roadmap item 4)
    "cost_mode": "tracking",  # "tracking", "contouring" or "hybrid", see `trajectory_cost`
    "path_window": 6.0,  # length in meters of the path window ahead of the current progress
    "path_behind": 0.5,  # length in meters of the path window behind the current progress
    "path_ds": 0.05,  # spacing in meters of the path window's points
    # Multi-modal MPPI and adaptive spread (roadmap item 5)
    "n_modes": 1,  # parallel means, the samples are split evenly between them
    "sigma_adapt": 0.0,  # gain of the adaptive noise spread per mode, 0 keeps sigma fixed
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
    w: Array,
    gate_mode: str = "slab",
    path: dict | None = None,
    cost_mode: str = "tracking",
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
        w: Weight of every state with shape (H + 1,), the length of its step in control periods.
            Running costs are weighted with it, so that a coarse step counts as much as the fine
            steps it replaces.
        gate_mode: How the gates are checked. "slab": every state within `gate_slab` of a gate's
            plane must be near its center. "crossing": every segment between two states that
            crosses a gate's plane must cross it near the center; this cannot be skipped by long
            steps. "both": the sum of the two.
        path: Window of the path around the current progress for the progress-based cost, see
            `contouring_cost`. Only used if `cost_mode` is not "tracking".
        cost_mode: "tracking": follow the time-indexed reference in position and velocity.
            "contouring": stay close to the path, make progress along it and stay below the speed
            profile, see `contouring_cost`. "hybrid": both.

    Returns:
        The scalar cost.
    """
    # Deviation from reference trajectory
    pos = X[:, :3]  # Extract positions from the state trajectory
    ref_pos = ref[:, :3]  # Extract reference positions
    tol = jnp.maximum(1e-6, cc["deviation_tolerance"])
    pos_error_per_step = jnp.square((pos - ref_pos) / tol)  # Compute per-step position error
    deviation_cost = jnp.sum(w[:, None] * pos_error_per_step)

    # Deviation from the reference velocity, so the speed profile (e.g. slowing before gates) is followed
    vel_tol = jnp.maximum(1e-6, cc["velocity_tolerance"])
    vel_error_per_step = jnp.square((X[:, 6:9] - ref[:, 3:6]) / vel_tol)
    velocity_cost = jnp.sum(w[:, None] * vel_error_per_step)

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
    gate_cost = 0.0
    if gate_mode in ("slab", "both"):
        slab_cost = jnp.where(frame_mask, distance_further_than_tolerance**2, 0)
        gate_cost += jnp.sum(w[:, None] * slab_cost)
    if gate_mode in ("crossing", "both"):
        # segments whose ends lie on different sides of a gate's plane, shape (H, n_gates)
        d0, d1 = distance[:-1], distance[1:]
        crosses = d0 * d1 < 0
        frac = d0 / jnp.where(crosses, d0 - d1, 1.0)  # where along the segment the plane is hit
        hit = gate_offset[:-1] + frac[..., None] * (gate_offset[1:] - gate_offset[:-1])
        hit_h = jnp.sum(hit * jnp.cross(gate_normal, jnp.array([0, 0, 1]))[None, :, :], axis=-1)
        hit_max = jnp.maximum(jnp.abs(hit[..., 2]), jnp.abs(hit_h))
        on_frame = crosses & (hit_max < cc["gate_frame_limit"])
        miss = jnp.maximum(hit_max / cc["gate_tolerance"] - 1, 0) ** 2
        # a crossing replaces the ~4 states that the slab checks at 2.5 m/s and 50 Hz
        gate_cost += cc["gate_crossing_scale"] * jnp.sum(jnp.where(on_frame, miss, 0))

    # Cost for coming close to an obstacle: tube around each pole, checked in the x-y plane
    obstacle_xy_distance = jnp.linalg.norm(pos[:, None, :2] - obstacles_pos[None, :, :2], axis=-1)
    # 0 outside the tube radius, rising to 1 at the pole's axis
    tube_penetration = jnp.maximum(1 - obstacle_xy_distance / cc["obstacle_radius"], 0)
    obstacle_cost = jnp.sum(w[:, None] * tube_penetration**2)

    total_cost = (
        cw["smoothness_weight"] * smoothness_cost
        + cw["gate_weight"] * gate_cost
        + cw["obstacle_weight"] * obstacle_cost
    )
    if cost_mode in ("tracking", "hybrid"):
        total_cost = (
            cw["deviation_weight"] * deviation_cost
            + cw["velocity_weight"] * velocity_cost
            + total_cost
        )
    if cost_mode in ("contouring", "hybrid"):
        total_cost += contouring_cost(X, path, cc, cw, w)

    return total_cost


PROJ_BEHIND = 3  # path points behind the previous state's point searched for the next state
PROJ_AHEAD = 13  # path points ahead of it, i.e. 0.65 m reach per step at 0.05 m spacing


def contouring_cost(X: Array, path: dict, cc: dict, cw: dict, w: Array) -> Array:
    """Progress-based cost: stay near the path, advance along it, pass the next gate.

    The path is a window of points sampled densely along the reference path around the current
    progress. The states are projected one after the other, each onto the nearest of the path
    points just behind or ahead of the previous state's point, and then along that point's tangent.

    Terms:
        contour: squared distance to the path over `contour_tolerance`, weighted per state.
        progress: reward for the progress of the last state beyond the current progress, in m.
        speed cap: squared speed above the mid-level speed profile at the state's progress. The
            profile slows down before unrevealed gates and in tight turns; as a cap instead of a
            target, the drone may go slower but not faster.
        gate bonus: reward for crossing the next gate's plane in its passing direction within
            `gate_bonus_radius` of its center. A sparse reward alone fails because most rollouts do
            not reach the next gate within the horizon, so it complements the dense progress term.

    Args:
        X: State trajectory with shape (H + 1, 13).
        path: Dict with `pts` (N, 3), `tangent` (N, 3) unit tangents, `s` (N,) progress, `v_cap`
            (N,) speed profile, `s0` () current progress, `i0` () index of the point at the current
            progress (where the projection of the first state starts), `gate` (6,) position and passing
            direction of the next gate, and `gate_valid` () 1 if there is a next gate, else 0.
        cc: Cost constants.
        cw: Cost weights.
        w: Weight of every state with shape (H + 1,).

    Returns:
        The scalar cost.
    """
    pos, vel = X[:, :3], X[:, 6:9]
    n_pts = path["pts"].shape[0]
    candidates = jnp.arange(PROJ_BEHIND + PROJ_AHEAD) - PROJ_BEHIND

    def project(i_prev: Array, p: Array) -> tuple[Array, Array]:
        # Search only near the previous state's path point, so the progress stays continuous and
        # cannot jump to a later section of the track that passes close by (shortcuts)
        idx = jnp.clip(i_prev + candidates, 0, n_pts - 1)
        i = idx[jnp.argmin(jnp.sum((path["pts"][idx] - p) ** 2, axis=-1))]
        return i, i

    _, j = jax.lax.scan(project, path["i0"], pos)
    offset = pos - path["pts"][j]
    along = jnp.sum(offset * path["tangent"][j], axis=-1)
    lateral = offset - along[:, None] * path["tangent"][j]
    s = path["s"][j] + along
    contour = jnp.sum(w * jnp.sum(lateral**2, axis=-1)) / cc["contour_tolerance"] ** 2
    progress = s[-1] - path["s0"]
    speed_excess = jnp.maximum(jnp.linalg.norm(vel, axis=-1) - path["v_cap"][j], 0.0)
    speed_cap = jnp.sum(w * speed_excess**2)

    # Bonus for crossing the next gate's plane forward near its center
    g_pos, g_dir = path["gate"][:3], path["gate"][3:]
    rel = pos - g_pos
    d = rel @ g_dir
    forward = (d[:-1] < 0) & (d[1:] >= 0)
    frac = d[:-1] / jnp.where(forward, d[:-1] - d[1:], 1.0)
    hit = rel[:-1] + frac[:, None] * (rel[1:] - rel[:-1])
    hit_lat = hit - (hit @ g_dir)[:, None] * g_dir  # component in the gate's plane
    inside = forward & (jnp.max(jnp.abs(hit_lat), axis=-1) < cc["gate_bonus_radius"])
    bonus = path["gate_valid"] * jnp.any(inside)

    return (
        cw["contour_weight"] * contour
        - cw["progress_weight"] * progress
        + cw["speed_cap_weight"] * speed_cap
        - cw["gate_bonus_weight"] * bonus
    )


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


NAN_COST = 1e9  # cost given to rollouts whose cost is NaN or inf


class MPPIOptions(NamedTuple):
    """Static options of the MPPI update. Changing one recompiles the update."""

    n_samples: int  # number of sampled control sequences
    noise_beta: float = 0.0  # AR(1) coefficient of the temporally smoothed noise, 0 for white noise
    truncated: bool = (
        False  # sample from a normal truncated to the action limits instead of clipping
    )
    include_nominal: bool = False  # always include the unperturbed nominal and a hover sequence
    nan_guard: bool = False  # replace NaN or inf rollout costs with NAN_COST
    integrator: str = "euler"  # see `dynamics_step`
    gate_mode: str = "slab"  # see `trajectory_cost`
    use_dist: bool = False  # add the estimated disturbance acceleration in the rollouts
    cost_mode: str = "tracking"  # see `trajectory_cost`
    n_modes: int = 1  # number of parallel means, see `mppi_update_multimodal`
    sigma_adapt: float = 0.0  # gain of the adaptive noise spread, 0 keeps sigma fixed


def sample_actions(
    key: Array, U_nom: Array, sigma: Array, lo: Array, hi: Array, opts: MPPIOptions
) -> Array:
    """Sample perturbed control sequences around `U_nom` within the action limits.

    White noise is i.i.d. normal per step. Smoothed noise is an AR(1) process per sample and channel,
    `e_t = beta * e_(t-1) + sqrt(1 - beta^2) * w_t`, which keeps the marginal standard deviation at
    `sigma` while correlating neighboring steps. Truncated sampling draws white noise from a normal
    truncated to the limits, so no probability mass piles up at the limits as with clipping. Smoothed
    noise is clipped, because a filtered truncated normal is not truncated anymore.

    Returns:
        The sampled sequences with shape (n_samples, H, 4).
    """
    shape = (opts.n_samples, *U_nom.shape)
    beta = opts.noise_beta
    if opts.truncated and beta == 0.0:
        safe_sigma = jnp.where(sigma > 0, sigma, 1.0)  # channels without noise (yaw) stay at U_nom
        lower, upper = (lo - U_nom) / safe_sigma, (hi - U_nom) / safe_sigma
        # Guard against a nominal sequence on or past a limit, where the interval would be empty
        lower, upper = jnp.minimum(lower, -1e-3), jnp.maximum(upper, 1e-3)
        noise = jax.random.truncated_normal(key, lower, upper, shape, dtype=jnp.float32)
    else:
        noise = jax.random.normal(key, shape, dtype=jnp.float32)
        if beta > 0.0:

            def ar1(prev: Array, w: Array) -> tuple[Array, Array]:
                e = beta * prev + jnp.sqrt(1 - beta**2) * w
                return e, e

            noise_t = jnp.swapaxes(noise, 0, 1)  # (H, n_samples, 4), scanned over time
            _, filtered = jax.lax.scan(ar1, noise_t[0], noise_t[1:])
            noise = jnp.swapaxes(jnp.concatenate((noise_t[:1], filtered)), 0, 1)
    return jnp.clip(U_nom + sigma * noise, lo, hi)


def mppi_update(
    key: Array,
    x0: Array,
    U_nom: Array,
    u_prev: Array,
    ref: Array,
    gates: Array,
    obstacles_pos: Array,
    params: dict,
    grid: dict,
    sigma: Array,
    target_ess: float,
    cc: dict,
    cw: dict,
    opts: MPPIOptions,
    dist_acc: Array | None = None,
    path: dict | None = None,
    sigma_cost: Array | None = None,
    return_extra: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array] | dict:
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
        grid: Time grid of the horizon: `dts` (H,) step lengths, `w` (H + 1,) cost weight of every
            state, and `disc`, the discretization of every step (see `discretize`).
        sigma: Per-channel standard deviation of the sampling noise with shape (4,).
        target_ess: Desired effective sample size as a fraction of n_samples.
        cc: Cost constants, see `COST_CONSTANTS`.
        cw: Cost weights, see `COST_WEIGHTS`.
        opts: Static options (shapes and feature switches), see `MPPIOptions`.
        dist_acc: Estimated disturbance acceleration with shape (3,), used if `opts.use_dist`.
        path: Path window for the contouring cost, see `contouring_cost`.
        sigma_cost: Noise scale that normalizes the smoothness cost, if it differs from the sampling
            `sigma` (adaptive spread). Defaults to `sigma`.
        return_extra: If True, return a dict that also holds the cost of the updated mean sequence
            (`mean_cost`) and the weighted spread of the samples per channel (`noise_std`).

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
    n_samples = opts.n_samples
    lo = jnp.array([-1.0, -1.0, -1.0, 4 * params["thrust_min"]])
    hi = jnp.array([1.0, 1.0, 1.0, 4 * params["thrust_max"]])
    sampled_actions = sample_actions(key, U_nom, sigma, lo, hi, opts)
    if opts.include_nominal:
        # The unperturbed nominal sequence and hovering are always candidates, so the update can
        # keep the current plan or stop if every perturbation is worse
        hover = (params["mass"] * -params["gravity_vec"][-1] - params["acc_coef"]) / params[
            "cmd_f_coef"
        ]
        hover_seq = jnp.zeros_like(U_nom).at[:, 3].set(hover)
        sampled_actions = sampled_actions.at[0].set(U_nom).at[1].set(hover_seq)

    # define batch functions
    dts, disc, w = grid["dts"], grid["disc"], grid["w"]

    dist = dist_acc if opts.use_dist else None

    def rollout_one(U: Array) -> Array:
        return rollout(x0, U, params, dts, disc, opts.integrator, dist)

    rollout_batch_of_trajectories = jax.vmap(rollout_one)

    def cost_one(X: Array, U: Array) -> Array:
        return trajectory_cost(
            X,
            U,
            u_prev,
            ref,
            gates,
            obstacles_pos,
            sigma if sigma_cost is None else sigma_cost,
            cc,
            cw,
            w,
            opts.gate_mode,
            path,
            opts.cost_mode,
        )

    trajectory_cost_for_batch = jax.vmap(cost_one)

    rollouts = rollout_batch_of_trajectories(sampled_actions)

    costs = trajectory_cost_for_batch(rollouts, sampled_actions)
    if opts.nan_guard:
        # A diverged rollout must not poison the softmin: give it a large finite cost instead
        costs = jnp.where(jnp.isfinite(costs), costs, NAN_COST)

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
    mean_rollout = rollout_one(U_new)
    mean_path = mean_rollout[:, :3]
    if not return_extra:
        return U_new, ess, lam, vis_pos, vis_weights, mean_path
    # Spread of the weighted samples per channel, for the adaptive noise
    noise_std = jnp.sqrt(jnp.sum(weights[:, None] * jnp.mean((sampled_actions - U_nom) ** 2, 1), 0))
    return {
        "U_new": U_new,
        "ess": ess,
        "lam": lam,
        "vis_pos": vis_pos,
        "vis_weights": vis_weights,
        "mean_path": mean_path,
        "mean_cost": cost_one(mean_rollout, U_new),
        "noise_std": noise_std,
    }


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
        # Time grid: `horizon` steps of 1 / freq, then `n_coarse` steps of `coarse_factor` / freq
        factors = np.array([1] * int(p["horizon"]) + [int(p["coarse_factor"])] * int(p["n_coarse"]))
        self._dts = time_grid(
            self._dt, int(p["horizon"]), int(p["n_coarse"]), int(p["coarse_factor"])
        )
        self._horizon = len(self._dts)  # number of planning steps
        # Index of every knot in the reference sampled at the control rate, shape (H + 1,)
        self._knot_idx = np.concatenate(([0], np.cumsum(factors)))
        w = np.concatenate(([1], factors if p["step_weighting"] == "dt" else 1 + 0 * factors))
        if p["discount_tau"] > 0:
            t_knots = self._dt * self._knot_idx  # time of every state
            w = w * np.exp(-t_knots / p["discount_tau"])
        self._grid = {
            "dts": jnp.asarray(self._dts),
            "w": jnp.asarray(w, dtype=jnp.float32),
            "disc": None,  # set below, once the drone parameters are loaded
        }
        self._n_samples = int(p["n_samples"])  # sampled control sequences per step
        self._sigma = jnp.asarray(p["sigma"], dtype=jnp.float32)  # noise std: r, p, y, thrust
        # fraction of samples that should effectively contribute to the update
        self._target_ess = float(p["target_ess"])
        self._step_ms = []  # wall time of every MPPI update, for benchmarking
        self._lam = float("nan")  # softmin temperature, adapted every step to reach the target ESS
        self._params = load_params(Dynamics.so_rpy_rotor_drag, config.sim.drone)
        self._grid["disc"] = discretize(self._params, self._dts)
        # Constant inputs of the update live on the device, so they are not copied every step
        self._params_dev = jax.device_put(
            {k: jnp.asarray(v, dtype=jnp.float32) for k, v in self._params.items()}
        )
        self._grid = jax.device_put(self._grid)
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

        # Shapes and feature switches are static for jit
        self._opts = MPPIOptions(
            n_samples=self._n_samples,
            noise_beta=float(p["noise_beta"]),
            truncated=bool(p["truncated"]),
            include_nominal=bool(p["include_nominal"]),
            nan_guard=bool(p["nan_guard"]),
            integrator=str(p["integrator"]),
            gate_mode=str(p["gate_mode"]),
            use_dist=float(p["dist_alpha"]) > 0,
            cost_mode=str(p["cost_mode"]),
            n_modes=int(p["n_modes"]),
            sigma_adapt=float(p["sigma_adapt"]),
        )
        # Several modes or an adaptive spread use the multi-modal update, which keeps one nominal
        # sequence and one noise spread per mode
        self._multimodal = self._opts.n_modes > 1 or self._opts.sigma_adapt > 0
        self._update_mm = jax.jit(mppi_update_multimodal, static_argnames=("opts",))
        self._shift_modes = jax.jit(jax.vmap(shift_controls, in_axes=(0, None, None)))
        self._mode = 0  # index of the executed mode
        n_window = int(round((p["path_window"] + p["path_behind"]) / p["path_ds"])) + 1
        self._path_offsets = p["path_ds"] * np.arange(n_window) - p["path_behind"]
        # Disturbance estimate and the one-step prediction it is updated from
        self._dist_alpha = float(p["dist_alpha"])
        self._dist_max = float(p["dist_max"])
        self._dist_axes = np.asarray(p["dist_axes"], dtype=np.float32)
        self._dist = np.zeros(3, dtype=np.float32)
        self._dist_log = []  # estimate at every step, for diagnostics
        self._vel_pred = None  # velocity predicted for the next observation
        # The ground holds the drone before takeoff, which the model does not know about, so the
        # estimate is only updated once the drone is airborne
        self._z_takeoff = float(obs["pos"][2]) + 0.05

        def predict_vel(x: Array, u: Array, dist_acc: Array) -> Array:
            disc0 = jax.tree.map(lambda a: a[0], self._grid["disc"])
            dt0 = self._grid["dts"][0]
            return dynamics_step(x, u, self._params, dt0, disc0, self._opts.integrator, dist_acc)[
                6:9
            ]

        self._predict_vel = jax.jit(predict_vel)
        self._update = jax.jit(mppi_update, static_argnames=("opts",))
        self._key = jax.random.key(int(p["rng_seed"]))
        # Warm start: hold hover thrust with level attitude
        self._U_nom = jnp.tile(jnp.array([0.0, 0.0, 0.0, self._hover_thrust]), (self._horizon, 1))
        self._u_prev = self._U_nom[0]
        self._U_modes = jnp.tile(self._U_nom, (self._opts.n_modes, 1, 1))
        self._sigma_modes = jnp.tile(self._sigma, (self._opts.n_modes, 1))
        # The thrust is not observed, so it is estimated by passing the commands through the lag
        self._thrust = 0.0
        self._ess = float("nan")
        self._vis = None  # (rollout positions, rollout weights, mean path), set in compute_control
        self._finished = False
        self._gate_now = None  # (positions, quaternions) of the currently perceived gates
        self._gate_old = []  # per gate, the poses that were perceived before the current one
        self._obstacle_now = None  # positions of the currently perceived obstacles
        self._obstacle_old = []  # per obstacle, the positions perceived before the current one

        self._ref = self._reference(obs)

        # Warm-up call so the jit compilation happens here and not in the first control step
        jax.block_until_ready(self._run_update(self._key, obs))

    def _run_update(self, key: Array, obs: dict[str, NDArray[np.floating]]) -> tuple:
        """Run one jitted MPPI update for the current observation and nominal sequence."""
        if self._multimodal:
            out = self._update_mm(
                key,
                self._state(obs),
                self._U_modes,
                self._sigma_modes,
                self._u_prev,
                self._ref,
                self._get_gates_pos_and_normal(obs),
                jnp.asarray(obs["obstacles_pos"]),
                self._params_dev,
                self._grid,
                self._sigma,
                self._target_ess,
                self._cc,
                self._cw,
                opts=self._opts,
                dist_acc=jnp.asarray(self._dist),
                path=self._path_window(obs) if self._opts.cost_mode != "tracking" else None,
            )
            U_modes, sigma_modes, best, *rest = out
            return (U_modes[best], *rest, (U_modes, sigma_modes, best))
        return self._update(
            key,
            self._state(obs),
            self._U_nom,
            self._u_prev,
            self._ref,
            self._get_gates_pos_and_normal(obs),
            jnp.asarray(obs["obstacles_pos"]),
            self._params_dev,
            self._grid,
            self._sigma,
            self._target_ess,
            self._cc,
            self._cw,
            opts=self._opts,
            dist_acc=jnp.asarray(self._dist),
            path=self._path_window(obs) if self._opts.cost_mode != "tracking" else None,
        )

    def _reference(self, obs: dict[str, NDArray[np.floating]]) -> NDArray:
        """Sample the mid-level reference at the knots of the time grid, shape (H + 1, 6)."""
        ref = self._mlc.get_trajectory(obs, int(self._knot_idx[-1]), self._dt)
        return ref[self._knot_idx]

    def _path_window(self, obs: dict[str, NDArray[np.floating]]) -> dict:
        """Sample the reference path and its speed profile around the current progress."""
        ref = self._mlc._planner._current_reference
        s0 = float(self._mlc._current_progress)
        s = np.clip(s0 + self._path_offsets, 0.0, ref.length)
        pts = ref.path(s)
        tangent = ref.path(s, 1)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
        profile = self._mlc._current_velocity_profile
        v_cap = profile[np.clip(np.searchsorted(ref.path.x, s), 0, len(profile) - 1)]
        n_passed = int(obs["n_gates_passed"])
        n_gates = len(obs["gate_sequence"])
        gate_valid = n_passed < n_gates
        gate = np.zeros(6)
        if gate_valid:
            i = int(obs["gate_sequence"][n_passed])
            normal = R.from_quat(obs["gates_quat"][i]).apply([1.0, 0.0, 0.0])
            gate = np.concatenate(
                (obs["gates_pos"][i], normal * obs["gate_sequence_direction"][n_passed])
            )
        f32 = jnp.float32
        return {
            "pts": jnp.asarray(pts, dtype=f32),
            "tangent": jnp.asarray(tangent, dtype=f32),
            "s": jnp.asarray(s, dtype=f32),
            "v_cap": jnp.asarray(v_cap, dtype=f32),
            "s0": jnp.asarray(s0, dtype=f32),
            "i0": jnp.asarray(int(np.argmin(np.abs(s - s0))), dtype=jnp.int32),
            "gate": jnp.asarray(gate, dtype=f32),
            "gate_valid": jnp.asarray(float(gate_valid), dtype=f32),
        }

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

        self._ref = self._reference(obs)
        self._track_gate_perception(obs)
        self._track_obstacle_perception(obs)
        self._update_disturbance(obs)

        # Fresh noise every step: split the key, keep one half and use the other
        self._key, subkey = jax.random.split(self._key)
        t_start = time.perf_counter()
        out = jax.block_until_ready(self._run_update(subkey, obs))
        U_new, ess, lam, vis_pos, vis_weights, mean_path = out[:6]
        if self._multimodal:
            U_modes, self._sigma_modes, best = out[6]
            self._mode = int(best)
            self._U_modes = self._shift_modes(U_modes, self._grid["dts"], self._dt)
        self._step_ms.append(1e3 * (time.perf_counter() - t_start))
        self._ess = float(ess)
        self._lam = float(lam)
        self._vis = (np.asarray(vis_pos), np.asarray(vis_weights), np.asarray(mean_path))

        self._u_prev = U_new[0]
        # Shift by one control period for the next call (on a uniform grid: drop the first input
        # and repeat the last one)
        self._U_nom = shift_controls(U_new, self._grid["dts"], self._dt)
        # Only the first input is applied, the rest is re-planned at the next step
        action = np.asarray(U_new[0], dtype=np.float32)
        # Propagate the thrust estimate with the same first-order lag the model uses
        x_now = self._state(obs)  # with the thrust estimate before this command's lag is applied
        thrust_dyn_coef = float(self._params["thrust_dyn_coef"])
        self._thrust += thrust_dyn_coef * (action[3] - self._thrust) * self._dt
        if self._dist_alpha > 0:
            self._vel_pred = np.asarray(
                self._predict_vel(x_now, jnp.asarray(action), jnp.asarray(self._dist))
            )
        return action

    def _update_disturbance(self, obs: dict[str, NDArray[np.floating]]):
        """Low-pass filter the velocity prediction error into a constant disturbance acceleration.

        The prediction of the last step already contains the current estimate, so its error is the
        part of the disturbance that is not yet explained: `d += alpha * (v_obs - v_pred) / dt`.
        """
        if self._dist_alpha <= 0 or self._vel_pred is None or obs["pos"][2] < self._z_takeoff:
            return
        residual = (np.asarray(obs["vel"]) - self._vel_pred) / self._dt
        self._dist = self._dist + self._dist_alpha * self._dist_axes * residual
        self._dist = np.clip(self._dist, -self._dist_max, self._dist_max).astype(np.float32)
        self._dist_log.append(self._dist.copy())

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
        self._U_modes = jnp.tile(self._U_nom, (self._opts.n_modes, 1, 1))
        self._sigma_modes = jnp.tile(self._sigma, (self._opts.n_modes, 1))
        self._thrust = 0.0
        self._dist = np.zeros(3, dtype=np.float32)
        self._vel_pred = None

    def _get_gates_pos_and_normal(self, obs: dict[str, NDArray[np.floating]]) -> Array:
        """Return the position and the normal vector of every gate with shape (n_gates, 6)."""
        # The gate normal is the gate's local x-axis in the world frame
        gates_normal = R.from_quat(obs["gates_quat"]).apply([1, 0, 0])
        return jnp.hstack([obs["gates_pos"], gates_normal])


def mppi_update_multimodal(
    key: Array,
    x0: Array,
    U_modes: Array,
    sigma_modes: Array,
    u_prev: Array,
    ref: Array,
    gates: Array,
    obstacles_pos: Array,
    params: dict,
    grid: dict,
    sigma0: Array,
    target_ess: float,
    cc: dict,
    cw: dict,
    opts: MPPIOptions,
    dist_acc: Array | None = None,
    path: dict | None = None,
) -> tuple:
    """One MPPI iteration with several modes (parallel means) and adaptive noise spread.

    The samples are split evenly between the modes. Every mode is updated like a single MPPI with
    its own temperature, and the mode whose updated mean sequence has the lowest cost is executed.
    Modes that converge to the same local optimum would waste samples, so the most expensive mode
    is reseeded every step from the best mean plus smooth noise. This keeps alternatives such as
    the other side of an obstacle alive instead of averaging both sides into the obstacle.

    With `opts.sigma_adapt > 0` the noise spread of every mode follows the weighted spread of its
    own samples (low-pass filtered with gain `sigma_adapt` and clipped to `[0.3, 2] * sigma0`).

    Args:
        key: JAX PRNG key.
        x0: Current state with shape (13,).
        U_modes: Nominal sequences of the modes with shape (K, H, 4).
        sigma_modes: Noise spread of the modes with shape (K, 4).
        u_prev: Previously applied input with shape (4,).
        ref: Reference at the knots with shape (H + 1, 6).
        gates: Position and normal of every gate with shape (n_gates, 6).
        obstacles_pos: Obstacle positions with shape (n_obstacles, 3).
        params: Drone parameters.
        grid: Time grid, see `mppi_update`.
        sigma0: Base noise spread with shape (4,), for the smoothness cost and the reseeding.
        target_ess: Desired effective sample size per mode as a fraction of its samples.
        cc: Cost constants.
        cw: Cost weights.
        opts: Static options; `n_samples` is the total over all modes.
        dist_acc: Estimated disturbance acceleration with shape (3,).
        path: Path window for the contouring cost.

    Returns:
        U_new_modes (K, H, 4), sigma_modes (K, 4), the index of the executed mode, and the ESS,
        temperature, visualized rollouts, their weights and the mean path of the executed mode.
    """
    n_modes = U_modes.shape[0]
    keys = jax.random.split(key, n_modes + 1)
    sub_opts = opts._replace(n_samples=opts.n_samples // n_modes)

    def one_mode(k: Array, U: Array, sig: Array) -> dict:
        return mppi_update(
            k,
            x0,
            U,
            u_prev,
            ref,
            gates,
            obstacles_pos,
            params,
            grid,
            sig,
            target_ess,
            cc,
            cw,
            opts=sub_opts,
            dist_acc=dist_acc,
            path=path,
            sigma_cost=sigma0,
            return_extra=True,
        )

    out = jax.vmap(one_mode)(keys[:n_modes], U_modes, sigma_modes)
    cost = jnp.where(jnp.isfinite(out["mean_cost"]), out["mean_cost"], NAN_COST)
    best = jnp.argmin(cost)
    U_new = out["U_new"]
    if opts.sigma_adapt > 0:
        a = opts.sigma_adapt
        sigma_modes = jnp.clip(
            (1 - a) * sigma_modes + a * out["noise_std"], 0.3 * sigma0, 2 * sigma0
        )
    if n_modes > 1:
        worst = jnp.argmax(cost)
        lo = jnp.array([-1.0, -1.0, -1.0, 4 * params["thrust_min"]])
        hi = jnp.array([1.0, 1.0, 1.0, 4 * params["thrust_max"]])
        seed_opts = opts._replace(n_samples=1, noise_beta=0.9, truncated=False)
        seed = sample_actions(keys[-1], U_new[best], sigma0, lo, hi, seed_opts)[0]
        U_new = U_new.at[worst].set(jnp.where(worst == best, U_new[worst], seed))
        sigma_modes = sigma_modes.at[worst].set(
            jnp.where(worst == best, sigma_modes[worst], sigma0)
        )
    return (
        U_new,
        sigma_modes,
        best,
        out["ess"][best],
        out["lam"][best],
        out["vis_pos"][best],
        out["vis_weights"][best],
        out["mean_path"][best],
    )
