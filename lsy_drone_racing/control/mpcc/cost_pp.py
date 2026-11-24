"""MPCC++ cost, after Krinner, Romero et al., "MPCC++", RSS 2024 (Eq. 6, 7 and 14).

Compared to the MPCC cost of `mpcc/cost.py`, the contour weight is constant and low, and safety at
the gates comes from a soft tunnel constraint instead (see `mpcc/corridor.py`). The contour error
is split into its horizontal and vertical components along the tunnel's axes. The progress speed
has no hard limits: the quadratic cost R_v * v^2 against the reward mu * v makes mu / (2 R_v) the
preferred progress speed.

The inputs follow the conventions of `mpcc/cost.py`: command differences replace the single rotor
thrust differences, and rpy rates replace the body rates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from lsy_drone_racing.control.mpcc.path import path_lookup, profile_lookup
from lsy_drone_racing.control.utils.collision import GATE_INNER_HALF_WIDTH
from lsy_drone_racing.control.utils.utils import drone_collision_radius

if TYPE_CHECKING:
    from crazyflow import Sim
    from jax import Array
    from numpy.typing import NDArray

    from lsy_drone_racing.control.mpcc.path import ArcLengthPath
    from lsy_drone_racing.control.planner.centerline import CenterlinePlanner

MPCC_PP_WEIGHTS = {
    # Lag weight, high so that theta stays a good estimate of the closest point
    "q_lag": 1000.0,
    # Contour weights along the tunnel's horizontal and upward axes. Low, since the tunnel and not
    # the contour cost keeps the drone clear of the gates
    "q_contour_h": 50.0,
    "q_contour_v": 100.0,
    "q_rates_rp": 0.01,  # Weight for roll and pitch rates
    "q_rates_yaw": 0.01,  # Weight for the yaw rate
    "r_du": jnp.array([1.0, 1.0, 1.0, 1.0]),  # Weights for command changes per channel
    "r_v_theta": 2.0,  # Weight for the squared progress speed
    "mu": 40.0,  # Reward for progress speed
    "w_tunnel": 1000.0,  # Weight of the tunnel barrier
    "alpha": 100.0,  # Slope of the tunnel barrier in 1/m
}

# Tunnel shape, see `corridor.tunnel_half_width_profile`
TUNNEL_MARGIN = 0.03  # Clearance in meters to the gate opening's edge, beyond the drone's radius
TUNNEL_W_NOMINAL = 1.0  # Half-width in meters away from gates
# The half-width is within 1 % of its value at the gate up to 0.1 m from the gate, which covers the
# drone's radius plus the frame's depth, and back to nominal about 0.6 m from the gate
TUNNEL_D_MID = 0.35
TUNNEL_STEEPNESS = 0.05

# Path points between drawn tunnel cross-sections. Every drawn segment uses one of the viewer's 1000
# geoms, and the rollouts already use most of them.
RENDER_STRIDE = 64


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
        cost_inputs: "weights" (see `MPCC_PP_WEIGHTS`) and "half_width", the tunnel's half-width on
            the path's grid with shape (N,).

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
    progress_cost = jnp.sum(weights["r_v_theta"] * v_theta**2 - weights["mu"] * v_theta)
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

    # No hard limits on the progress except that it never reverses
    limits = ProgressLimits(v_max=jnp.inf, dv_max=jnp.inf, v_slow=jnp.inf, slack=jnp.inf)

    def __init__(self, config: dict, planner: CenterlinePlanner):
        """Initialize the objective.

        Args:
            config: The race configuration.
            planner: Planner of the centerline, which provides the gate crossings.
        """
        self._planner = planner
        self._w_gate = (
            GATE_INNER_HALF_WIDTH - drone_collision_radius(config.sim.drone) - TUNNEL_MARGIN
        )
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
            TUNNEL_W_NOMINAL,
            TUNNEL_D_MID,
            TUNNEL_STEEPNESS,
        )
        # No speed profile. The largest float instead of inf, since interpolating inf gives NaN.
        self.speed_limit = jnp.full(path.positions.shape[0], jnp.finfo(jnp.float32).max)

    def cost_inputs(self, obs: dict[str, NDArray[np.floating]]) -> dict:
        """The weights and the tunnel's half-width."""
        return {"weights": MPCC_PP_WEIGHTS, "half_width": self._half_width}

    def render(self, sim: Sim):
        """Draw the tunnel's four edges."""
        corners = tunnel_corners(
            self._path.positions[::RENDER_STRIDE],
            self._path.tangents[::RENDER_STRIDE],
            self._half_width[::RENDER_STRIDE],
        )
        for edge in np.asarray(corners):
            draw_line(sim, edge, rgba=(0.3, 0.5, 1.0, 0.5), start_size=1.5, end_size=1.5)
