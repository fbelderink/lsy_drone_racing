"""MPPI contouring controller (`mpcc/mppi_mpcc.py`) following a minimum-time PMM path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mpcc.mppi_mpcc import MPPIMPCC
from lsy_drone_racing.control.planner.pmm import PMMPlanner

if TYPE_CHECKING:
    import numpy as np
    from crazyflow import Sim
    from numpy.typing import NDArray


class AttitudeMPPIMPCCPMM(Controller):
    """MPPI on the contouring cost, following the path of `PMMPlanner`."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._mpcc = MPPIMPCC(obs, config, PMMPlanner(obs, config))

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next roll, pitch, yaw and collective thrust command.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The command [roll, pitch, yaw, thrust] as a numpy array.
        """
        return self._mpcc.compute_control(obs)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Check whether the path's end is reached or all gates are passed.

        Returns:
            True if the controller is finished, False otherwise.
        """
        return self._mpcc.is_finished(obs)

    def render_callback(self, sim: Sim):
        """Draw the planned path, the sampled rollouts and the mean path."""
        self._mpcc.render(sim)
