import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import crazyflow
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.utils.types import ObservationDict, State


def gate_poses_in_sequence(obs: ObservationDict) -> tuple[NDArray, NDArray]:
    """Extract sequence of gate poses along track from observation dict."""
    gates_pos = obs["gates_pos"][obs["gate_sequence"]]
    gates_normals = R.from_quat(obs["gates_quat"][obs["gate_sequence"]]).apply([1.0, 0.0, 0.0])
    gates_normals *= obs["gate_sequence_direction"][:, None]

    return gates_pos, gates_normals


def obs_to_state(obs: ObservationDict) -> State:
    pos = obs["pos"]
    vel = obs["vel"]
    rpy = R.from_quat(obs["quat"]).as_euler("xyz")

    # ang_vel = #TODO

    state = np.concat([pos, rpy, vel], axis=-1).reshape(-1)

    return state


def obs_to_gate_state(obs: ObservationDict):

    gates_pos = obs["gates_pos"]
    gates_rpy = R.from_quat(obs["gates_quat"]).as_euler("xyz")

    state = np.concat([gates_pos, gates_rpy], axis=-1).reshape(-1)

    return state


def drone_collision_radius(drone: str) -> float:
    """Radius of the drone's collision sphere from crazyflow's MuJoCo model."""
    xml_path = Path(crazyflow.__file__).parent / "drones" / f"{drone}.xml"
    geom = ET.parse(xml_path).getroot().find(".//geom[@name='col_sphere']")
    return float(geom.get("size").split()[0])


def obstacle_radius() -> float:
    """Radius of the obstacle poles from the environment's MuJoCo model."""
    xml_path = Path(__file__).parents[2] / "envs" / "assets" / "obstacle.xml"
    geom = ET.parse(xml_path).getroot().find(".//geom[@name='obstacle']")
    return float(geom.get("size").split()[0])


class GateGeometry(NamedTuple):
    """Collision geometry of a gate's frame, a square ring in the gate's plane."""

    inner: float  # Half-width of the opening
    outer: float  # Half-width of the frame's outer edge
    half_depth: float  # Half-thickness of the frame along the gate's normal


def gate_geometry() -> GateGeometry:
    """Frame geometry of the gates from the environment's MuJoCo model.

    The frame collides as four bars around the opening. The top bar, a box centered above the
    opening, gives all sizes: its x size is the half-depth, y the outer half-width, and its lower
    edge (center z minus z size) the half-width of the opening.
    """
    xml_path = Path(__file__).parents[2] / "envs" / "assets" / "gate.xml"
    geom = ET.parse(xml_path).getroot().find(".//geom[@name='col_top']")
    center_z = float(geom.get("pos").split()[2])
    half_depth, outer, half_height = (float(v) for v in geom.get("size").split())
    return GateGeometry(inner=center_z - half_height, outer=outer, half_depth=half_depth)
