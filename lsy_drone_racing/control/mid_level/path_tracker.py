"""Tracking of the drone's progress on the high level planner's path.

The path tracker follows the high level planner's reference path and the drone's progress s on
it. It is all a contouring controller (MPCC) needs, which chooses the speed itself: around the
progress, it provides a window of the path in which the controller measures the progress and the
distance to the path of its rollouts. Mid level controllers that also decide the speed along the
path build on it, see `MidLevelController`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.planner.high_level_planner import SplinePlanner
    from lsy_drone_racing.control.planner.reference import ProgressReference
    from lsy_drone_racing.control.utils.types import ObservationDict


class PathTracker:
    """Follows the planner's reference path and the drone's progress on it."""

    def __init__(
        self,
        planner: SplinePlanner,
        window_speed: float = 10.0,
        window_behind: float = 0.1,
        window_ahead: float = 1.0,
        window_ds: float = 0.05,
    ):
        """Initialize the tracker.

        Args:
            planner: The high level planner that provides the reference path.
            window_speed: Speed in m/s that sets the reach, the distance covered at this speed
                over the horizon, which the path window's length is given in.
            window_behind: Length of the path window behind the drone's progress, in multiples
                of the reach.
            window_ahead: Length of the path window ahead of the drone's progress, in multiples
                of the reach.
            window_ds: Spacing in meters of the path window's points.
        """
        self._planner = planner
        self.window_speed = window_speed
        self.window_behind = window_behind
        self.window_ahead = window_ahead
        self.window_ds = window_ds

        self._reference: ProgressReference | None = None
        self._progress = 0.0
        self._offsets: NDArray | None = None  # (P,) offsets of the path window from the progress
        self._reach: float | None = None  # reach the offsets were computed for

    @property
    def reference(self) -> ProgressReference:
        """The reference path."""
        return self._reference

    @property
    def progress(self) -> float:
        """The drone's progress s in meters on the reference path."""
        return self._progress

    def update(self, obs: ObservationDict, horizon: int, dt: float) -> bool:
        """Follow the planner's path and the drone's progress on it.

        Args:
            obs: The current observation, passed on to the high level planner.
            horizon: Number of steps of the low level controller's horizon.
            dt: Time step in seconds of the horizon.

        Returns:
            Whether the path changed.
        """
        ref, updated = self._planner.plan(obs)
        updated = updated or self._reference is None
        self._reference = ref
        if updated:
            self._on_new_reference(ref, obs)

        reach = self.window_speed * horizon * dt
        if reach != self._reach:
            self._reach = reach
            self._offsets = np.arange(
                -self.window_behind * reach, self.window_ahead * reach + 1e-9, self.window_ds
            )

        # predict the progress by integrating the speed along the path, then correct it by
        # projecting the position onto the path around the prediction
        tangent = ref(self._progress, nu=1)
        tangent = tangent / max(la.norm(tangent), 1e-9)
        s_pred = np.clip(self._progress + dt * np.dot(obs["vel"], tangent), 0.0, ref.length)
        self._progress = float(ref.project(obs["pos"], s_pred=s_pred))
        return updated

    def path_window(self) -> NDArray:
        """The reference path around the drone's progress, as of the last `update`.

        The window reaches from window_behind to window_ahead times the reach around the
        progress, clipped to the path, with a fixed number of points so that jitted users do not
        recompile.

        Returns:
            (P, 4) positions and their progress relative to the drone's progress in meters.
        """
        ref = self._reference
        s = np.clip(self._progress + self._offsets, 0.0, ref.length)
        return np.column_stack([ref(s), s - self._progress])

    def _on_new_reference(self, ref: ProgressReference, obs: ObservationDict):
        """Called by `update` whenever the planner provides a new path; nothing to do here."""
