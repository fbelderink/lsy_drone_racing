"""Minimum-time path through the gates from the point-mass-model (PMM) planner.

The planner itself (Teissing et al., "Real-Time Planning of Minimum-Time Trajectories for Agile UAV
Flight", RA-L 2024) is C++ code in `third_party/pmm_uav_planner`. It runs as the separate program
`tools/pmm_cli` (built by `tools/setup_pmm.sh`), which reads the waypoints from stdin and writes the
sampled trajectory to stdout. Only the trajectory's positions are used: the follower decides the
speed along the path itself.

Every gate becomes three waypoints along its normal (before, center, after) so that the path
crosses it roughly straight. When the next gate lies behind a gate, an extra turnaround waypoint
leads the path around the gate's frame instead of back through it. Where the planned path still
comes too close to a gate or an obstacle, a waypoint pushed out of the way is inserted and the
path is planned again (`planner/path_repair.py`).
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la
from crazyflow.dynamics import Dynamics, load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.planner.path_repair import (
    first_collision,
    insert_repair_waypoints,
    insertion_index,
    repair_path_hit,
    repair_waypoint,
)
from lsy_drone_racing.control.utils.collision import GATE_OUTER_HALF_WIDTH
from lsy_drone_racing.control.utils.utils import drone_collision_radius, obstacle_radius

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

PMM_CLI_PATH = Path(__file__).parents[3] / "tools" / "pmm_cli" / "build" / "pmm_cli"
CLI_TIMEOUT = 1.0  # Seconds before a planner call is aborted

GATE_OFFSET = 0.3  # Distance in meters of the before/after waypoints from a gate's center
END_DISTANCE = 0.7  # Distance in meters of the path's end point behind the last gate
END_SPEED = 4.0  # Speed in m/s at the path's end, along the last gate's normal
ACC_FACTOR = 1.0  # Fraction of the drone's maximum thrust acceleration the planner may use
MAX_VEL_NORM = 11.0  # Speed limit in m/s
SAMPLE_DT = 0.005  # Sampling period in seconds of the planned trajectory
PATH_DS = 0.01  # Spacing in meters of the dense path points
TURN_EXIT = 0.5  # Distance in meters of a turnaround waypoint behind the gate's plane
# Distance in meters before the next gate's plane within which its center waypoint is dropped too:
# a path planned from the drone through a center only a few centimeters ahead loops at its start
CENTER_SKIP = 0.1
TURN_CLEARANCE = 0.15  # Distance in meters between a turnaround waypoint and the drone at the frame
MAX_REPAIRS = 12  # Maximum number of waypoints inserted to repair collisions of one plan


def gate_normals(gates_quat: NDArray[np.floating]) -> NDArray[np.floating]:
    """Unit normals of the gates, the gates' local x-axes in the world frame.

    Args:
        gates_quat: Gate orientations as xyzw quaternions with shape (n_gates, 4).

    Returns:
        The normals with shape (n_gates, 3).
    """
    return R.from_quat(gates_quat).apply([1.0, 0.0, 0.0])


def turnaround_waypoint(
    center: NDArray[np.floating],
    direction: NDArray[np.floating],
    next_center: NDArray[np.floating],
    side_offset: float,
) -> NDArray[np.floating]:
    """Waypoint behind a gate from which the path turns back around the gate's frame.

    It lies TURN_EXIT behind the gate's plane, moved by side_offset to the left, to the right or up,
    whichever points most towards the next gate. Never down, where the gate's stand is.

    Args:
        center: Gate center with shape (3,).
        direction: Crossing direction of the gate, a unit vector with shape (3,).
        next_center: Center of the next gate with shape (3,).
        side_offset: Distance in meters from the gate's center to the side, beyond the frame.

    Returns:
        The waypoint with shape (3,).
    """
    horizontal = np.cross(direction, [0.0, 0.0, 1.0])
    horizontal /= la.norm(horizontal)
    sides = np.array([[0.0, 0.0, 1.0], horizontal, -horizontal])
    side = sides[np.argmax(sides @ (next_center - center))]
    return center + TURN_EXIT * direction + side_offset * side


def gate_waypoints(
    pos: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_normal: NDArray[np.floating],
    gate_sequence: NDArray[np.integer],
    gate_direction: NDArray[np.integer],
    n_passed: int,
    turn_side_offset: float,
    offset: float = GATE_OFFSET,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Waypoints before, at and after every gate that is still to be passed.

    The before-waypoint of the next gate is dropped when the drone is already closer to the gate's
    plane than it, so the path does not turn back, and so is its center waypoint when the drone is
    within CENTER_SKIP of the plane. When the following gate lies behind a gate's
    plane, a turnaround waypoint is added after it, see `turnaround_waypoint`.

    Args:
        pos: Drone position with shape (3,).
        gates_pos: Gate centers with shape (n_gates, 3).
        gates_normal: Gate normals with shape (n_gates, 3).
        gate_sequence: 0-based gate ids in the order they are passed, with shape (n_sequence,).
        gate_direction: +1 to pass a gate along its normal, -1 against it, with shape (n_sequence,).
        n_passed: Number of gates of the sequence already passed, smaller than n_sequence.
        turn_side_offset: Sideways distance in meters of turnaround waypoints from the center.
        offset: Distance in meters of the before/after waypoints from the gate's center.

    Returns:
        - Waypoints with shape (n_waypoints, 3).
        - Crossing direction of the last gate, a unit vector with shape (3,).
    """
    ids = gate_sequence[n_passed:]
    centers = gates_pos[ids]
    directions = gates_normal[ids] * gate_direction[n_passed:, None]
    waypoints = []
    for i, (center, direction) in enumerate(zip(centers, directions)):
        waypoints += [center - offset * direction, center, center + offset * direction]
        is_last = i + 1 == len(centers)
        if not is_last and np.dot(centers[i + 1] - center, direction) < 0:
            next_center = centers[i + 1]
            waypoints.append(turnaround_waypoint(center, direction, next_center, turn_side_offset))
    waypoints = np.array(waypoints)
    distance_to_plane = -np.dot(pos - centers[0], directions[0])
    n_dropped = int(distance_to_plane < offset) + int(distance_to_plane < CENTER_SKIP)
    waypoints = waypoints[n_dropped:]
    if n_passed > 0:
        # Right after a gate, keep its turnaround waypoint, or the path turns back through its frame
        passed = gate_sequence[n_passed - 1]
        passed_direction = gates_normal[passed] * gate_direction[n_passed - 1]
        behind_passed = np.dot(pos - gates_pos[passed], passed_direction)
        next_is_behind = np.dot(centers[0] - gates_pos[passed], passed_direction) < 0
        if next_is_behind and behind_passed < TURN_EXIT:
            turn = turnaround_waypoint(
                gates_pos[passed], passed_direction, centers[0], turn_side_offset
            )
            waypoints = np.vstack((turn, waypoints))
    return waypoints, directions[-1]


def max_thrust_acceleration(drone: str) -> float:
    """Maximum collective thrust acceleration of the drone at its nominal mass.

    Args:
        drone: Crazyflow drone model name, e.g. "cf21B_500".

    Returns:
        The acceleration in m/s^2, gravity not included.
    """
    params = load_params(Dynamics.so_rpy, drone)
    return float(4 * params["thrust_max"] / params["mass"])


def format_cli_input(
    start_pos: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    waypoints: NDArray[np.floating],
    end_vel: NDArray[np.floating],
    gravity: float,
    max_acc_norm: float,
    max_vel_norm: float = MAX_VEL_NORM,
    sample_dt: float = SAMPLE_DT,
) -> str:
    """Write a planning problem in the input format of `pmm_cli`.

    Args:
        start_pos: Start position with shape (3,).
        start_vel: Start velocity with shape (3,).
        waypoints: Waypoints after the start, the last one is the end point, with shape (n, 3).
        end_vel: Velocity at the end point with shape (3,).
        gravity: Gravitational acceleration in m/s^2, positive.
        max_acc_norm: Maximum thrust acceleration in m/s^2.
        max_vel_norm: Speed limit in m/s.
        sample_dt: Sampling period in seconds of the output.

    Returns:
        The text to pass to `pmm_cli` on stdin.
    """
    points = np.vstack((start_pos, waypoints))
    header = [gravity, max_acc_norm, max_vel_norm, sample_dt, *start_vel, *end_vel]
    lines = [" ".join(f"{x:.9g}" for x in header), str(len(points))]
    lines += [" ".join(f"{x:.9g}" for x in p) for p in points]
    return "\n".join(lines) + "\n"


def parse_cli_output(text: str) -> NDArray[np.floating]:
    """Read the trajectory written by `pmm_cli`.

    Args:
        text: The program's stdout.

    Returns:
        Samples [t, pos(3), vel(3)] with shape (n_samples, 7).

    Raises:
        ValueError: If the text is not at least two rows of 7 numbers.
    """
    samples = np.loadtxt(text.splitlines(), ndmin=2)
    if samples.shape[0] < 2 or samples.shape[1] != 7:
        raise ValueError(f"Expected at least 2 rows of 7 values, got shape {samples.shape}")
    return samples


def run_pmm(cli_input: str, cli_path: Path = PMM_CLI_PATH) -> NDArray[np.floating]:
    """Run `pmm_cli` on a planning problem.

    Args:
        cli_input: The problem, see `format_cli_input`.
        cli_path: Path of the `pmm_cli` executable.

    Returns:
        Samples [t, pos(3), vel(3)] with shape (n_samples, 7).

    Raises:
        RuntimeError: If the program is missing or fails.
    """
    if not cli_path.is_file():
        raise RuntimeError(
            f"PMM planner not found at {cli_path}. It is built by tools/setup_pmm.sh when the "
            "pixi environment is activated, see tools/pmm_cli/README.md."
        )
    result = subprocess.run(
        [str(cli_path)], input=cli_input, capture_output=True, text=True, timeout=CLI_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"PMM planner failed: {result.stderr.strip()}")
    return parse_cli_output(result.stdout)


class DensePathReference:
    """Path through densely sampled points, parameterised by arc length.

    Follows the `PathReference` interface of `planner/interface.py`.
    """

    def __init__(self, points: NDArray[np.floating], version: int = 1, ds: float = PATH_DS):
        """Build the path.

        Args:
            points: Points along the path with shape (n, 3). Repeated points are allowed.
            version: Version number, must differ from the previous path's.
            ds: Spacing in meters of the resampled points.
        """
        s = np.append(0.0, np.cumsum(la.norm(np.diff(points, axis=0), axis=1)))
        distinct = np.append(True, np.diff(s) > 1e-9)
        s, points = s[distinct], points[distinct]
        if len(s) < 2:
            raise ValueError("The path needs at least two distinct points.")

        # Resample evenly, so the spline does not oscillate between unevenly spaced points
        grid = np.linspace(0.0, s[-1], max(int(s[-1] / ds) + 1, 2))
        self._dense_knots = np.stack([np.interp(grid, s, points[:, i]) for i in range(3)], axis=1)
        s_dense = np.append(0.0, np.cumsum(la.norm(np.diff(self._dense_knots, axis=0), axis=1)))
        self.path = CubicSpline(s_dense, self._dense_knots, axis=0)
        self.length = float(s_dense[-1])
        self._version = version

    def get_progress(
        self,
        pos: NDArray[np.floating],
        prev_progress: float | None = None,
        search_range_s: tuple[float, float] = (0.3, 1.0),
    ) -> float:
        """Arc length of the path point closest to pos.

        Args:
            pos: Query position with shape (3,).
            prev_progress: If given, only the path within search_range_s around it is searched.
                This avoids jumping to other sections of the path, e.g. where it crosses itself.
            search_range_s: (behind, ahead) search window in meters around prev_progress.

        Returns:
            The progress s in meters.
        """
        s = self.path.x
        pts = self._dense_knots

        lo, hi = 0, len(s) - 1
        if prev_progress is not None:
            behind, ahead = search_range_s
            # Index of the segment containing the lower bound, so the result is not always ahead
            # of prev_progress
            lo = np.clip(
                np.searchsorted(s, prev_progress - behind, side="right") - 1, 0, len(s) - 2
            )
            hi = np.clip(
                np.searchsorted(s, prev_progress + ahead, side="right") - 1, lo + 1, len(s) - 1
            )

        # Project onto every segment of the dense polyline within the window
        a = pts[lo:hi]
        ab = pts[lo + 1 : hi + 1] - a
        t = np.sum((pos - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-12)
        t = np.clip(t, 0.0, 1.0)
        dist = la.norm(a + t[:, None] * ab - pos, axis=1)

        i = np.argmin(dist)
        return float(s[lo + i] + t[i] * (s[lo + i + 1] - s[lo + i]))


class PMMPlanner:
    """Plans a minimum-time path through the gates, avoiding gate frames and obstacles.

    Re-plans from the drone's current state when a gate's or an obstacle's pose is revealed.

    Follows the `PathPlanner` interface of `planner/interface.py`.
    """

    starts_at_drone = True

    def __init__(self, obs: dict[str, NDArray[np.floating]], config: dict):
        """Plan the initial path.

        Args:
            obs: The initial observation of the environment.
            config: The race configuration.
        """
        params = load_params(Dynamics.so_rpy, config.sim.drone)
        self._gravity = float(-params["gravity_vec"][-1])
        self._max_acc_norm = ACC_FACTOR * max_thrust_acceleration(config.sim.drone)
        self._drone_radius = drone_collision_radius(config.sim.drone)
        self._obstacle_radius = obstacle_radius()
        self._gates_visited = np.array(obs["gates_visited"], copy=True)
        self._obstacles_visited = np.array(obs["obstacles_visited"], copy=True)
        self._reference = self._replan(obs, version=1)

    def plan(self, obs: dict[str, NDArray[np.floating]]) -> DensePathReference:
        """Return the path, re-planned from the current state if a new pose was revealed.

        Args:
            obs: The current observation of the environment.

        Returns:
            The current path.
        """
        revealed = np.any(obs["gates_visited"] & ~self._gates_visited) or np.any(
            obs["obstacles_visited"] & ~self._obstacles_visited
        )
        all_passed = int(obs["n_gates_passed"]) >= len(obs["gate_sequence"])
        if revealed and not all_passed:
            self._gates_visited = np.array(obs["gates_visited"], copy=True)
            self._obstacles_visited = np.array(obs["obstacles_visited"], copy=True)
            self._reference = self._replan(obs, version=self._reference._version + 1)
        return self._reference

    def _replan(self, obs: dict[str, NDArray[np.floating]], version: int) -> DensePathReference:
        """Plan from the drone's current position and velocity through the remaining gates."""
        start = time.perf_counter()
        waypoints, last_direction = gate_waypoints(
            obs["pos"],
            obs["gates_pos"],
            gate_normals(obs["gates_quat"]),
            obs["gate_sequence"],
            obs["gate_sequence_direction"],
            int(obs["n_gates_passed"]),
            GATE_OUTER_HALF_WIDTH + self._drone_radius + TURN_CLEARANCE,
        )
        waypoints = np.vstack((waypoints, waypoints[-1] + END_DISTANCE * last_direction))
        gates = np.hstack((obs["gates_pos"], gate_normals(obs["gates_quat"])))
        # Obstacles are not inflated by their position uncertainty: their true position is revealed
        # 0.7 m before the drone reaches them, which triggers a replan
        collision_args = (
            gates,
            obs["obstacles_pos"],
            np.zeros(len(obs["obstacles_pos"])),
            self._drone_radius,
            self._obstacle_radius,
        )
        # Waypoints that collide themselves cannot be fixed by inserting others next to them
        waypoints = np.array([repair_waypoint(w, *collision_args) for w in waypoints])

        for n_repairs in range(MAX_REPAIRS + 1):
            cli_input = format_cli_input(
                obs["pos"],
                obs["vel"],
                waypoints,
                END_SPEED * last_direction,
                self._gravity,
                self._max_acc_norm,
            )
            samples = run_pmm(cli_input)
            points = samples[:, 1:4]
            hit = first_collision(points, *collision_args)
            if hit is None or n_repairs == MAX_REPAIRS:
                break
            index = insertion_index(points, waypoints, hit)
            new_waypoints = repair_path_hit(points, hit, *collision_args)
            waypoints = insert_repair_waypoints(waypoints, index, new_waypoints)
        if hit is not None:
            logger.warning("PMM path v%d still collides after %d repairs", version, MAX_REPAIRS)

        reference = DensePathReference(points, version=version)
        logger.debug(
            "PMM path v%d: %.2f m, %.2f s, %d repairs, planned in %.1f ms",
            version,
            reference.length,
            samples[-1, 0],
            n_repairs,
            1e3 * (time.perf_counter() - start),
        )
        return reference
