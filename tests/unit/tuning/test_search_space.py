import numpy as np
import pytest

from lsy_drone_racing.control.mpcc.cost_pp import (
    MPCC_PP_COLLISION_WEIGHTS,
    MPCC_PP_WEIGHTS,
    TUNNEL_W_NOMINAL,
)
from lsy_drone_racing.tuning.search_space import (
    SEARCH_SPACE,
    default_values,
    unit_to_values,
    values_to_overrides,
    values_to_unit,
)


@pytest.mark.unit
def test_unit_cube_corners_map_to_range_ends():
    dim = len(SEARCH_SPACE)
    low = unit_to_values(np.zeros(dim))
    high = unit_to_values(np.ones(dim))
    for bound in SEARCH_SPACE:
        assert low[bound.name] == pytest.approx(bound.low)
        assert high[bound.name] == pytest.approx(bound.high)


@pytest.mark.unit
def test_cube_center_maps_to_geometric_or_arithmetic_mean():
    center = unit_to_values(np.full(len(SEARCH_SPACE), 0.5))
    for bound in SEARCH_SPACE:
        if bound.log:
            expected = np.sqrt(bound.low * bound.high)
        else:
            expected = 0.5 * (bound.low + bound.high)
        assert center[bound.name] == pytest.approx(expected)


@pytest.mark.unit
def test_defaults_round_trip_to_todays_overrides():
    x = values_to_unit(default_values())
    assert np.all((x >= 0.0) & (x <= 1.0))
    overrides = values_to_overrides(unit_to_values(x))
    for name in ("q_lag", "q_contour_h", "q_contour_v", "q_rates_rp", "mu", "r_v_theta"):
        assert overrides[name] == pytest.approx(MPCC_PP_WEIGHTS[name])
    assert overrides["r_v_theta_unrevealed"] == pytest.approx(
        MPCC_PP_WEIGHTS["r_v_theta_unrevealed"]
    )
    np.testing.assert_allclose(overrides["r_du"], MPCC_PP_WEIGHTS["r_du"])
    assert overrides["tunnel_w_nominal"] == pytest.approx(TUNNEL_W_NOMINAL)
    assert overrides["w_obs"] == pytest.approx(MPCC_PP_COLLISION_WEIGHTS["w_obs"])
    assert overrides["w_crash"] == pytest.approx(MPCC_PP_COLLISION_WEIGHTS["w_crash"])


@pytest.mark.unit
def test_speeds_set_progress_speed_weights():
    values = default_values() | {"mu": 60.0, "v_pref": 15.0, "v_slow": 3.0}
    overrides = values_to_overrides(values)
    r_v_theta = overrides["r_v_theta"]
    assert r_v_theta == pytest.approx(60.0 / (2 * 15.0))
    # Preferred speed near an unrevealed gate: mu / (2 * (r_v_theta + r_v_theta_unrevealed))
    assert 60.0 / (2 * (r_v_theta + overrides["r_v_theta_unrevealed"])) == pytest.approx(3.0)


@pytest.mark.unit
def test_slow_speed_above_preferred_speed_gives_no_extra_weight():
    values = default_values() | {"v_pref": 3.0, "v_slow": 5.0}
    assert values_to_overrides(values)["r_v_theta_unrevealed"] == 0.0
