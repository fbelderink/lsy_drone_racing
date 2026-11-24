"""Generic Interface for high level planners."""

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

    def _handle_gates_update(self, obs: ObservationDict) -> bool:
        """Adjust reference with updated gate positions."""
        mask = obs["gates_visited"][obs["gate_sequence"]]

        gates_pos, gates_normals = gate_poses_in_sequence(obs)
        # use new center point (could optimize)
        p = gates_pos
        v = gates_normals
        pts, pt_mask = self._current_reference.get_current_gate_positions(mask)

        # check if any gates positions changed
        if np.all(pts[pt_mask] == p[mask]):
            return False

        self._current_reference = self._current_reference.update_gates(p, v, mask)

        if mask[-1] and not np.equal(pts[pt_mask][-1], p[-1]).all():
            # final gate was revealed
            final_p = gates_pos[-1] + self._overshoot * gates_normals[-1]

            self._current_reference = self._current_reference.update_final_waypoint(
                final_p=final_p,
                final_v=self._tangent_scale * gates_normals[-1],
                final_a=np.zeros_like(gates_pos[-1]),
            )

        # extra waypoints need to be fully rebuild
        self._current_reference = self._current_reference.without_extras()

        return True

    def _avoid_collisions(self, obs: ObservationDict):
        self._ca_strat.avoid_collisions(obs, self._current_reference)

    def plan(self, obs: ObservationDict) -> tuple[ProgressReference, bool]:
        """Plan a path through course from given observations."""
        updated = False
        gates_pos, gates_normals = gate_poses_in_sequence(obs)

        if self._current_reference is None:
            # initial path

            self._current_reference = ProgressReference.initial_reference(
                obs["pos"], gates_pos, gates_normals, tangent_scale=self._tangent_scale
            )

            return self._current_reference, True

        updated |= self._handle_gates_update(obs)
        # if updated:  # || discovered new obstacle
        self._current_reference = self._ca_strat.avoid_collisions(obs, self._current_reference)

        return self._current_reference, updated
