import numpy as np
import pytest

from lsy_drone_racing.control.mpcc.cost_pp import (
    MPCC_PP_COLLISION_WEIGHTS,
    MPCC_PP_WEIGHTS,
    TUNNEL_W_NOMINAL,
    MPCCPPParams,
)
from lsy_drone_racing.control.mpcc.mppi_mpcc import COLLISION_WEIGHTS


@pytest.mark.unit
def test_no_overrides_keeps_todays_settings():
    params = MPCCPPParams.from_overrides({})
    assert params.weights.keys() == MPCC_PP_WEIGHTS.keys()
    for name, value in MPCC_PP_WEIGHTS.items():
        np.testing.assert_array_equal(params.weights[name], value)
    assert params.tunnel_w_nominal == TUNNEL_W_NOMINAL
    assert params.collision_weights == MPCC_PP_COLLISION_WEIGHTS


@pytest.mark.unit
def test_mpcc_pp_collision_weights_differ_only_in_tuned_entries():
    # The shared defaults of the other MPCC controllers are not changed
    assert COLLISION_WEIGHTS["w_obs"] == 30.0
    assert COLLISION_WEIGHTS["w_crash"] == 3000.0
    assert MPCC_PP_COLLISION_WEIGHTS.keys() == COLLISION_WEIGHTS.keys()
    for name in COLLISION_WEIGHTS.keys() - {"w_obs", "w_crash"}:
        assert MPCC_PP_COLLISION_WEIGHTS[name] == COLLISION_WEIGHTS[name]


@pytest.mark.unit
def test_override_replaces_only_its_key():
    mu_default = MPCC_PP_WEIGHTS["mu"]
    params = MPCCPPParams.from_overrides(
        {
            "mu": 80.0,
            "r_du": [2.0, 2.0, 2.0, 2.0],
            "tunnel_w_nominal": 0.5,
            "w_obs": 300.0,
            "w_crash": 1000.0,
        }
    )
    assert params.weights["mu"] == 80.0
    np.testing.assert_array_equal(params.weights["r_du"], [2.0, 2.0, 2.0, 2.0])
    assert params.weights["r_du"].dtype == MPCC_PP_WEIGHTS["r_du"].dtype
    assert params.weights["q_lag"] == MPCC_PP_WEIGHTS["q_lag"]
    assert params.tunnel_w_nominal == 0.5
    assert params.collision_weights["w_obs"] == 300.0
    assert params.collision_weights["w_crash"] == 1000.0
    assert params.collision_weights["obstacle_sigma"] == COLLISION_WEIGHTS["obstacle_sigma"]
    # The module defaults are not modified
    assert MPCC_PP_WEIGHTS["mu"] == mu_default
    assert MPCC_PP_COLLISION_WEIGHTS["w_obs"] != 300.0


@pytest.mark.unit
def test_unknown_override_raises():
    with pytest.raises(KeyError, match="horizon"):
        MPCCPPParams.from_overrides({"horizon": 30})
