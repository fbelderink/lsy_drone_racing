import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.utils.collision import gate_contact, gates_have_stand

DRONE_RADIUS = 0.086
# Lower gate 0.7 m up, facing +x
LOWER_GATE = [0.0, 0.0, 0.7, 1.0, 0.0, 0.0]


def gate(pos: list[float], yaw: float) -> list[float]:
    """Gate row [pos, normal] for a gate at pos turned by yaw about the z-axis."""
    return [*pos, np.cos(yaw), np.sin(yaw), 0.0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("upper", "has_stand"),
    [
        (gate([0.0, 0.0, 1.42], 0.0), False),  # stacked
        (gate([0.1, 0.05, 1.42], np.pi + 0.1), False),  # stacked, flipped, both poses randomized
        (gate([0.5, 0.0, 1.42], 0.0), True),  # beside the lower gate
        (gate([0.0, 0.0, 1.42], np.pi / 2), True),  # turned against the lower gate
    ],
)
def test_gates_have_stand(upper: list[float], has_stand: bool):
    gates = np.array([LOWER_GATE, upper])
    assert gates_have_stand(gates).tolist() == [True, has_stand]


@pytest.mark.unit
@pytest.mark.parametrize("xp", [np, jnp])
def test_lower_gate_opening_is_free_below_a_stacked_gate(xp: object):
    gates = xp.asarray([LOWER_GATE, gate([0.0, 0.0, 1.42], 0.0)])
    in_lower_opening = xp.asarray([[0.0, 0.0, 0.7]])
    assert not bool(gate_contact(in_lower_opening, gates, DRONE_RADIUS)[0])


@pytest.mark.unit
def test_single_gate_keeps_its_stand():
    below_gate = np.array([[0.0, 0.0, 0.2]])
    assert bool(gate_contact(below_gate, np.array([LOWER_GATE]), DRONE_RADIUS)[0])
