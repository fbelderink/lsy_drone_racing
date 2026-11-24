"""This file contains types used throughout the repository."""

from typing import NewType

import numpy as np
from numpy.typing import NDArray

ObservationDict = NewType("ObservationDict", dict[str, NDArray[np.floating]])
State = NewType("State", NDArray)

Trajectory = NewType("Trajectory", NDArray[np.floating])  # (H, 6) [x, y, z, vx, vy, vz]

StateCommand = NewType("StateCommand", NDArray[np.floating])  # (16, ) [x, y, z, vx, ..., wz]
AttitudeCommand = NewType(
    "AttitudeCommand", NDArray[np.floating]
)  # (4, ) [thrust, roll, pitch, yaw]
