"""Interface between path planners and the controllers that follow their paths.

The attribute names match the existing `Reference` class, so `SplinePlanner` satisfies these
protocols without changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline


class PathReference(Protocol):
    """A geometric path parameterised by arc length s in meters."""

    path: CubicSpline  # path(s, nu) gives the position (nu=0) or its nu-th derivative in s
    length: float  # Total arc length in meters
    _version: int  # Changes whenever the path changes

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
            search_range_s: (behind, ahead) search window in meters around prev_progress.

        Returns:
            The progress s in meters.
        """
        ...


class PathPlanner(Protocol):
    """Plans a path through the gates and updates it when new gate or obstacle poses are sensed."""

    # Whether every new path starts at the drone's current position, so a controller can restart
    # its progress at zero instead of projecting the old progress onto the new path
    starts_at_drone: bool

    def plan(self, obs: dict[str, NDArray[np.floating]]) -> PathReference:
        """Return the path to follow, re-planned if the observation revealed new poses.

        Args:
            obs: The current observation of the environment.

        Returns:
            The current path.
        """
        ...
