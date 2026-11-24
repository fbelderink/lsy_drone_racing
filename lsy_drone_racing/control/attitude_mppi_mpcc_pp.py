"""MPCC++ controller: MPPI (`mpcc/mppi_mpcc.py`) on the MPCC++ cost (`mpcc/cost_pp.py`).

The drone follows a centerline through the gate centers (`planner/centerline.py`) within a tunnel
that narrows at the gates (`mpcc/corridor.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mpcc.cost_pp import CorridorObjective
from lsy_drone_racing.control.mpcc.mppi_mpcc import MPPIMPCC
from lsy_drone_racing.control.planner.centerline import CenterlinePlanner

if TYPE_CHECKING:
    import numpy as np
    from crazyflow import Sim
    from numpy.typing import NDArray


class AttitudeMPPIMPCCPP(Controller):
    """MPPI on the MPCC++ cost, following the centerline of `CenterlinePlanner`."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        planner = CenterlinePlanner(obs)
        self._mpcc = MPPIMPCC(obs, config, planner, CorridorObjective(config, planner))

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
        """Draw the centerline, the tunnel, the sampled rollouts and the mean path."""
        self._mpcc.render(sim)
