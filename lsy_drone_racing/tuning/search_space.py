"""Search space of the MPCC++ tuning: the unit cube mapped to controller parameters.

TuRBO works on the unit cube [0, 1]^d. Every dimension maps to one search variable, on a log scale
for weights. Two search variables are speeds instead of weights: the preferred progress speed
mu / (2 r_v_theta) and the preferred progress speed near an unrevealed gate, since the weights
themselves are tightly coupled to mu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.mpcc.cost_pp import (
    MPCC_PP_COLLISION_WEIGHTS,
    MPCC_PP_WEIGHTS,
    TUNNEL_W_NOMINAL,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Bound:
    """Range of one search variable.

    Attributes:
        name: Name of the search variable.
        low: Lower end of the range.
        high: Upper end of the range.
        log: Whether the unit interval maps to the range on a log scale.
    """

    name: str
    low: float
    high: float
    log: bool


SEARCH_SPACE = (
    Bound("q_lag", 100.0, 10_000.0, log=True),
    Bound("q_contour_h", 5.0, 500.0, log=True),
    Bound("q_contour_v", 10.0, 1000.0, log=True),
    Bound("q_rates_rp", 1e-3, 1.0, log=True),
    Bound("r_du_scale", 0.1, 10.0, log=True),  # Weight of all four command changes
    Bound("mu", 10.0, 400.0, log=True),
    Bound("v_pref", 3.0, 20.0, log=False),  # Preferred progress speed in m/s
    Bound("v_slow", 1.0, 5.0, log=False),  # Preferred progress speed near unrevealed gates in m/s
    Bound("tunnel_w_nominal", 0.3, 1.5, log=False),
    Bound("w_obs", 10.0, 3000.0, log=True),
    Bound("w_crash", 300.0, 30_000.0, log=True),
)


def unit_to_values(x: NDArray[np.floating]) -> dict[str, float]:
    """Map a point of the unit cube to the search variables.

    Args:
        x: Point in [0, 1]^d with shape (d,), ordered as `SEARCH_SPACE`.

    Returns:
        The search variables by name.
    """
    values = {}
    for bound, u in zip(SEARCH_SPACE, np.asarray(x, dtype=float), strict=True):
        if bound.log:
            values[bound.name] = float(bound.low * (bound.high / bound.low) ** u)
        else:
            values[bound.name] = float(bound.low + (bound.high - bound.low) * u)
    return values


def values_to_unit(values: dict[str, float]) -> NDArray[np.floating]:
    """Map the search variables to a point of the unit cube, the inverse of `unit_to_values`.

    Args:
        values: The search variables by name.

    Returns:
        The point with shape (d,), outside [0, 1] for values outside the ranges.
    """
    x = []
    for bound in SEARCH_SPACE:
        value = values[bound.name]
        if bound.log:
            x.append(np.log(value / bound.low) / np.log(bound.high / bound.low))
        else:
            x.append((value - bound.low) / (bound.high - bound.low))
    return np.array(x)


def values_to_overrides(values: dict[str, float]) -> dict:
    """Convert the search variables to controller overrides, see `MPCCPPParams.from_overrides`.

    The progress speed weights follow from the preferred speeds, which are where the progress
    reward mu * v balances the cost r * v^2: v = mu / (2 r).

    Args:
        values: The search variables by name.

    Returns:
        The overrides by name, JSON-serializable.
    """
    mu = values["mu"]
    r_v_theta = mu / (2 * values["v_pref"])
    # No extra weight if the slow speed is above the preferred speed
    r_v_theta_unrevealed = max(0.0, mu / (2 * values["v_slow"]) - r_v_theta)
    return {
        "q_lag": values["q_lag"],
        "q_contour_h": values["q_contour_h"],
        "q_contour_v": values["q_contour_v"],
        "q_rates_rp": values["q_rates_rp"],
        "r_du": [values["r_du_scale"]] * 4,
        "mu": mu,
        "r_v_theta": r_v_theta,
        "r_v_theta_unrevealed": r_v_theta_unrevealed,
        "tunnel_w_nominal": values["tunnel_w_nominal"],
        "w_obs": values["w_obs"],
        "w_crash": values["w_crash"],
    }


def default_values() -> dict[str, float]:
    """The search variables of the controller's current defaults.

    Returns:
        The search variables by name.
    """
    weights = MPCC_PP_WEIGHTS
    mu = weights["mu"]
    r_v_theta = weights["r_v_theta"]
    return {
        "q_lag": weights["q_lag"],
        "q_contour_h": weights["q_contour_h"],
        "q_contour_v": weights["q_contour_v"],
        "q_rates_rp": weights["q_rates_rp"],
        # All four channels have the same weight
        "r_du_scale": float(weights["r_du"][0]),
        "mu": mu,
        "v_pref": mu / (2 * r_v_theta),
        "v_slow": mu / (2 * (r_v_theta + weights["r_v_theta_unrevealed"])),
        "tunnel_w_nominal": TUNNEL_W_NOMINAL,
        "w_obs": MPCC_PP_COLLISION_WEIGHTS["w_obs"],
        "w_crash": MPCC_PP_COLLISION_WEIGHTS["w_crash"],
    }
