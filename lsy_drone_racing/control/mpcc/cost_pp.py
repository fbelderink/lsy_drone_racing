"""MPCC++ cost, after Krinner, Romero et al., "MPCC++", RSS 2024 (Eq. 6, 7 and 14).

Compared to the MPCC cost of `mpcc/cost.py`, the contour weight is constant and low, and safety at
the gates comes from a soft tunnel constraint instead (see `mpcc/corridor.py`). The contour error
is split into its horizontal and vertical components along the tunnel's axes. The progress speed
has no hard upper limit: the quadratic cost R_v * v^2 against the reward mu * v makes mu / (2 R_v)
the preferred progress speed. R_v rises where the path approaches a gate whose true pose is not
revealed yet, so the drone slows down until it sees the gate, similar to how the contour weight of
`mpcc/cost.py` rises at the gates.

The inputs follow the conventions of `mpcc/cost.py`: command differences replace the single rotor
thrust differences, and rpy rates replace the body rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim.visualize import draw_line

from lsy_drone_racing.control.mpcc.corridor import (
    tunnel_barrier,
    tunnel_corners,
    tunnel_frame,
    tunnel_half_width_profile,
)
from lsy_drone_racing.control.mpcc.cost import ProgressLimits, contouring_errors
from lsy_drone_racing.control.mpcc.mppi_mpcc import COLLISION_WEIGHTS
from lsy_drone_racing.control.mpcc.path import path_lookup, profile_lookup
from lsy_drone_racing.control.utils.collision import GATE_INNER_HALF_WIDTH
from lsy_drone_racing.control.utils.utils import drone_collision_radius

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crazyflow import Sim
    from jax import Array
    from numpy.typing import NDArray

    from lsy_drone_racing.control.mpcc.path import ArcLengthPath
    from lsy_drone_racing.control.planner.centerline import CenterlinePlanner

# The tuned values below come from `scripts/tune_mpcc_pp.py` and are kept at full precision: the
# closed loop amplifies even rounding-level changes of the weights into different flights.
MPCC_PP_WEIGHTS = {
    # Lag weight, high so that theta stays a good estimate of the closest point (tuned)
    "q_lag": 2003.9231253985897,
    # Contour weights along the tunnel's horizontal and upward axes (tuned). Low, since the tunnel
    # and not the contour cost keeps the drone clear of the gates
    "q_contour_h": 40.22683513603317,
    "q_contour_v": 395.06866235912594,
    "q_rates_rp": 0.0061208242460775875,  # Weight for roll and pitch rates (tuned)
    # Weight for the yaw rate. Without yaw noise, all samples have the same yaw rate, so this weight
    # has no effect and is not tuned
    "q_rates_yaw": 0.01,
    # Weights for command changes per channel (tuned, the same for all channels)
    "r_du": jnp.full(4, 0.904801132215005),
    # Weight for the squared progress speed (tuned). The preferred progress speed is
    # mu / (2 * r_v_theta), about 7.4 m/s
    "r_v_theta": 7.836243719585712,
    # Additional weight for the squared progress speed near an unrevealed gate (tuned). The
    # preferred progress speed there is mu / (2 * (r_v_theta + r_v_theta_unrevealed)), about 2.4 m/s
    "r_v_theta_unrevealed": 16.87366843284105,
    # Distance in meters over which the additional weight fades in at `slow_radius` from the gate
    "slow_sigma": 0.1,
    "mu": 116.15527473906332,  # Reward for progress speed (tuned)
    "w_tunnel": 1000.0,  # Weight of the tunnel barrier
    "alpha": 100.0,  # Slope of the tunnel barrier in 1/m
}

# Tunnel shape, see `corridor.tunnel_half_width_profile`
TUNNEL_MARGIN = 0.03  # Clearance in meters to the gate opening's edge, beyond the drone's radius
TUNNEL_W_NOMINAL = 1.2892846530813704  # Half-width in meters away from gates (tuned)
# The half-width is within 1 % of its value at the gate up to 0.1 m from the gate, which covers the
# drone's radius plus the frame's depth, and back to nominal about 0.6 m from the gate
TUNNEL_D_MID = 0.35
TUNNEL_STEEPNESS = 0.05

# How far in meters past the next gate's crossing the measured progress may be, since the
# environment counts a gate as passed only after the step in which the drone crossed it
GATE_PASS_MARGIN = 0.1

# How far in meters beyond the sensor range around an unrevealed gate the drone already slows down
SLOW_MARGIN = 0.3

# Obstacle and crash cost weights of MPCC++, see `mppi_mpcc.COLLISION_WEIGHTS` (w_obs and w_crash
# tuned). The shared defaults stay unchanged for the other MPCC controllers.
MPCC_PP_COLLISION_WEIGHTS = COLLISION_WEIGHTS | {
    "w_obs": 17.927181046581076,
    "w_crash": 2972.7668810521463,
}


@dataclass(frozen=True)
class MPCCPPParams:
    """Tunable parameters of the MPCC++ controller, by default the module's constants.

    Attributes:
        weights: Cost weights, see `MPCC_PP_WEIGHTS`.
        tunnel_w_nominal: Tunnel half-width in meters away from gates.
        collision_weights: Obstacle and crash cost weights, see `MPCC_PP_COLLISION_WEIGHTS`.
    """

    weights: dict = field(default_factory=lambda: dict(MPCC_PP_WEIGHTS))
    tunnel_w_nominal: float = TUNNEL_W_NOMINAL
    collision_weights: dict = field(default_factory=lambda: dict(MPCC_PP_COLLISION_WEIGHTS))

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any]) -> MPCCPPParams:
        """The default parameters with some of them replaced.

        The cost weights keep the types of the defaults, so that new values do not recompile the
        jitted MPPI update. The collision weights are compile-time constants of `MPPIMPCC`.

        Args:
            overrides: New values by name: any key of `MPCC_PP_WEIGHTS` ("r_du" as a sequence of 4
                values), "tunnel_w_nominal", "w_obs" and "w_crash".

        Returns:
            The parameters.

        Raises:
            KeyError: If a name is none of the above.
        """
        weights = dict(MPCC_PP_WEIGHTS)
        collision_weights = dict(MPCC_PP_COLLISION_WEIGHTS)
        tunnel_w_nominal = TUNNEL_W_NOMINAL
        for name, value in overrides.items():
            if name == "tunnel_w_nominal":
                tunnel_w_nominal = float(value)
            elif name in ("w_obs", "w_crash"):
                collision_weights[name] = float(value)
            elif name == "r_du":
                weights["r_du"] = jnp.asarray(value, dtype=MPCC_PP_WEIGHTS["r_du"].dtype)
            elif name in weights:
                weights[name] = float(value)
            else:
                raise KeyError(f"Unknown MPCC++ parameter: {name}")
        return cls(weights, tunnel_w_nominal, collision_weights)


# Path points between drawn tunnel cross-sections. Every drawn segment uses one of the viewer's 1000
# geoms, and the rollouts already use most of them.
RENDER_STRIDE = 64


def progress_speed_weight(
    path_pos: Array, gates_pos: Array, gates_revealed: Array, slow_radius: float, weights: dict
) -> Array:
    """Weight of the squared progress speed, which rises near gates whose pose is not revealed.

    The weight depends on the path position, not on the drone position. It rises along a sigmoid of
    the horizontal distance to every unrevealed gate, which is halfway up at slow_radius, the same
    horizontal distance at which the sensors reveal the poses.

    Args:
        path_pos: Path positions with shape (..., 3).
        gates_pos: Gate positions with shape (n_gates, 3).
        gates_revealed: Whether each gate's true pose is known, with shape (n_gates,).
        slow_radius: Horizontal distance in meters from a gate at which the slowdown is halfway.
        weights: Cost weights, see `MPCC_PP_WEIGHTS`.

    Returns:
        The weights with shape (...,).
    """
    distance = jnp.linalg.norm(path_pos[..., None, :2] - gates_pos[:, :2], axis=-1)
    near = jax.nn.sigmoid((slow_radius - distance) / weights["slow_sigma"])
    n_near_unrevealed = jnp.sum(near * ~gates_revealed, axis=-1)
    return weights["r_v_theta"] + weights["r_v_theta_unrevealed"] * n_near_unrevealed


def mpcc_pp_cost(
    X: Array,
    U: Array,
    u_prev: Array,
    theta: Array,
    v_theta: Array,
    dv_theta: Array,
    path: ArcLengthPath,
    cost_inputs: dict,
) -> Array:
    """Compute the MPCC++ cost of a single rollout.

    Args:
        X: State trajectory with shape (H + 1, 13).
        U: Control sequence with shape (H, 4).
        u_prev: Previous control input with shape (4,).
        theta: Progress trajectory with shape (H + 1,).
        v_theta: Progress speed trajectory with shape (H + 1,).
        dv_theta: Progress acceleration sequence with shape (H,). Not penalised.
        path: The path to follow.
        cost_inputs: "weights" (see `MPCC_PP_WEIGHTS`), "half_width", the tunnel's half-width on
            the path's grid with shape (N,), and for the slowdown near unrevealed gates (see
            `progress_speed_weight`) "gates_pos", "gates_revealed" and "slow_radius".

    Returns:
        The scalar cost.
    """
    weights = cost_inputs["weights"]
    path_pos, tangent = path_lookup(path, theta)
    contour, lag = contouring_errors(X[:, :3], path_pos, tangent)
    n, b = tunnel_frame(tangent)
    # The contour error is normal to the tangent, so it is e_n * n + e_b * b
    e_n = jnp.sum(contour * n, axis=-1)
    e_b = jnp.sum(contour * b, axis=-1)
    half_width = profile_lookup(cost_inputs["half_width"], path.length, theta)

    horizontal_cost = weights["q_contour_h"] * jnp.sum(e_n**2)
    vertical_cost = weights["q_contour_v"] * jnp.sum(e_b**2)
    lag_cost = weights["q_lag"] * jnp.sum(lag**2)
    roll_pitch_rates_cost = weights["q_rates_rp"] * jnp.sum(X[:, 9:11] ** 2)
    yaw_rate_cost = weights["q_rates_yaw"] * jnp.sum(X[:, 11] ** 2)
    command_change = jnp.diff(jnp.vstack((u_prev, U)), axis=0)
    command_cost = jnp.sum(weights["r_du"] * command_change**2)
    r_v_theta = progress_speed_weight(
        path_pos,
        cost_inputs["gates_pos"],
        cost_inputs["gates_revealed"],
        cost_inputs["slow_radius"],
        weights,
    )
    progress_cost = jnp.sum(r_v_theta * v_theta**2 - weights["mu"] * v_theta)
    tunnel_cost = weights["w_tunnel"] * jnp.sum(
        tunnel_barrier(e_n, e_b, half_width, weights["alpha"])
    )

    return (
        horizontal_cost
        + vertical_cost
        + lag_cost
        + roll_pitch_rates_cost
        + yaw_rate_cost
        + command_cost
        + progress_cost
        + tunnel_cost
    )


class CorridorObjective:
    """MPCC++ objective for `MPPIMPCC`: the cost above with a tunnel around a centerline.

    The tunnel narrows at the gate crossings of `CenterlinePlanner`'s current centerline.
    """

    # The progress never reverses, and may start a step at most 1 m/s faster than the drone moves
    # along the path. Without that bound, the progress can run past a gate's narrow section while
    # the drone is still before the gate, and the drone then flies around the gate.
    limits = ProgressLimits(v_max=jnp.inf, dv_max=jnp.inf, v_slow=jnp.inf, slack=1.0)

    def __init__(
        self, config: dict, planner: CenterlinePlanner, params: MPCCPPParams | None = None
    ):
        """Initialize the objective.

        Args:
            config: The race configuration.
            planner: Planner of the centerline, which provides the gate crossings.
            params: Cost weights and tunnel width, the defaults if None.
        """
        self._params = MPCCPPParams() if params is None else params
        self._planner = planner
        self._w_gate = (
            GATE_INNER_HALF_WIDTH - drone_collision_radius(config.sim.drone) - TUNNEL_MARGIN
        )
        self._slow_radius = config.env.sensor_range + SLOW_MARGIN
        self._path = None
        self._half_width = None
        self.speed_limit = None

    cost_fn = staticmethod(mpcc_pp_cost)

    def on_new_path(self, path: ArcLengthPath):
        """Compute the tunnel's half-width on the new path's grid."""
        self._path = path
        self._half_width = tunnel_half_width_profile(
            path,
            self._planner.gate_progress,
            self._w_gate,
            self._params.tunnel_w_nominal,
            TUNNEL_D_MID,
            TUNNEL_STEEPNESS,
        )
        # No speed profile. The largest float instead of inf, since interpolating inf gives NaN.
        self.speed_limit = jnp.full(path.positions.shape[0], jnp.finfo(jnp.float32).max)

    def cost_inputs(self, obs: dict[str, NDArray[np.floating]]) -> dict:
        """The weights, the tunnel's half-width and the gates for the slowdown."""
        return {
            "weights": self._params.weights,
            "half_width": self._half_width,
            "gates_pos": jnp.asarray(obs["gates_pos"]),
            "gates_revealed": jnp.asarray(obs["gates_visited"]),
            "slow_radius": self._slow_radius,
        }

    def max_progress(self, obs: dict[str, NDArray[np.floating]]) -> float:
        """Progress just past the crossing of the next gate that is not passed yet.

        The drone cannot be further along the path before the environment counts the gate as
        passed. This keeps the measured progress from skipping a gate, e.g. where the path passes
        close to a later section of itself.
        """
        n_passed = int(obs["n_gates_passed"])
        if n_passed >= len(self._planner.gate_progress):
            return float("inf")
        return float(self._planner.gate_progress[n_passed]) + GATE_PASS_MARGIN

    def render(self, sim: Sim):
        """Draw the tunnel's four edges."""
        corners = tunnel_corners(
            self._path.positions[::RENDER_STRIDE],
            self._path.tangents[::RENDER_STRIDE],
            self._half_width[::RENDER_STRIDE],
        )
        for edge in np.asarray(corners):
            draw_line(sim, edge, rgba=(0.3, 0.5, 1.0, 0.5), start_size=1.5, end_size=1.5)
