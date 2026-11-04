from abc import abstractmethod

from lsy_drone_racing.control.utils.types import ObservationDict, Reference
from lsy_drone_racing.control.utils.utils import obs_to_gate_state


class HighLevelPlanner:
    def __init__(self, obs: ObservationDict, drone: str = "cf21B_500"):

        self._gate_state = obs_to_gate_state(obs)

        self._gate_sequence = obs["gate_sequence"]
        self._gate_sequence_dir = obs["gate_sequence_direction"]

        self._current_reference = None

    @abstractmethod
    def plan(self, obs: ObservationDict) -> Reference:
        """ """
        return NotImplementedError()
