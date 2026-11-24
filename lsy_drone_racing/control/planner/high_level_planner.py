"""High level planner: a spline path through the gates that avoids the known obstacles.

The planner starts from a reference through the nominal gate poses. Whenever the observation
reveals the exact pose of a gate or obstacle, it updates the gate knots (and the final waypoint
once the last gate is revealed), drops the old extra waypoints and pushes the path away from the
obstacles again. The returned flag tells the mid level controller when the path has changed.
"""

import numpy as np

from lsy_drone_racing.control.planner.reference import ProgressReference
from lsy_drone_racing.control.planner.strategies import CollisionAvoidanceStrategy
from lsy_drone_racing.control.utils.types import ObservationDict
from lsy_drone_racing.control.utils.utils import gate_poses_in_sequence


class SplinePlanner:
    """Spline Planner through the gates which pushes the path away from obstacles."""

    def __init__(self, obs: ObservationDict, config: dict):
        """Initialize high level spline-based planner."""
        self._overshoot = 0.5
        self._tangent_scale = 0.8
        self._current_reference = None

        self._ca_strat = CollisionAvoidanceStrategy(config)

        self._gates_visited = obs["gates_visited"]
        self._obstacles_visited = obs["obstacles_visited"]

    def _handle_gates_update(self, obs: ObservationDict):
        """Adjust reference with updated gate positions."""
        mask = obs["gates_visited"][obs["gate_sequence"]]

        gates_pos, gates_normals = gate_poses_in_sequence(obs)
        # use new center point (could optimize)
        p = gates_pos
        v = gates_normals

        self._current_reference = self._current_reference.update_gates(p, v, mask)

        if (~self._gates_visited[obs["gate_sequence"]] & mask)[-1]:
            # final gate was revealed
            final_p = gates_pos[-1] + self._overshoot * gates_normals[-1]

            self._current_reference = self._current_reference.update_final_waypoint(
                final_p=final_p,
                final_v=self._tangent_scale * gates_normals[-1],
                final_a=np.zeros_like(gates_pos[-1]),
            )

        # extra waypoints need to be fully rebuild
        self._current_reference = self._current_reference.without_extras()

    def _avoid_collisions(self, obs: ObservationDict):
        self._ca_strat.avoid_collisions(obs, self._current_reference)

    def plan(self, obs: ObservationDict) -> tuple[ProgressReference, bool]:
        """Plan a path through course from given observations."""
        gates_pos, gates_normals = gate_poses_in_sequence(obs)

        if self._current_reference is None:
            # initial path
            self._current_reference = ProgressReference.initial_reference(
                obs["pos"], gates_pos, gates_normals, tangent_scale=self._tangent_scale
            )

            return self._current_reference, True

        gates_update = (~self._gates_visited & obs["gates_visited"]).any()
        obstacles_update = (~self._obstacles_visited & obs["obstacles_visited"]).any()

        if gates_update:
            self._handle_gates_update(obs)

        if gates_update or obstacles_update:
            self._current_reference = self._ca_strat.avoid_collisions(obs, self._current_reference)

        self._gates_visited = obs["gates_visited"]
        self._obstacles_visited = obs["obstacles_visited"]

        return self._current_reference, gates_update or obstacles_update
