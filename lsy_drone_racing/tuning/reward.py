"""Reward of one tuning candidate over several seeded episodes.

After the MPCC++ paper (Krinner, Romero et al., RSS 2024, Sec. IV-C), with the share of failed
episodes in place of the solver failure rate:

    R = -T - FAILURE_PENALTY * f - MISSED_GATES_PENALTY * m

T is the mean lap time of the finished episodes, f the share of failed episodes (crash or
timeout), and m the share of gates missed, averaged over the failed episodes. m tells apart
candidates that fail every episode, which f alone rates the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Lap time in seconds if no episode finished: the episode limit of 1500 steps at 50 Hz
FAILED_LAP_TIME = 30.0
# Large enough that one failure among 10 episodes outweighs any lap time gain
FAILURE_PENALTY = 100.0
MISSED_GATES_PENALTY = 5.0


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one seeded episode.

    Attributes:
        seed: Seed of the environment.
        finished: Whether the drone passed all gates.
        gates_passed: Number of gates passed.
        n_gates: Number of gates of the track.
        lap_time: Time in seconds until the episode ended.
    """

    seed: int
    finished: bool
    gates_passed: int
    n_gates: int
    lap_time: float


def reward(results: Sequence[EpisodeResult]) -> float:
    """Reward of one candidate, higher is better.

    Args:
        results: Outcomes of the candidate's episodes, at least one.

    Returns:
        The reward.
    """
    finished = [result for result in results if result.finished]
    failed = [result for result in results if not result.finished]
    if finished:
        mean_lap_time = float(np.mean([result.lap_time for result in finished]))
    else:
        mean_lap_time = FAILED_LAP_TIME
    failure_share = len(failed) / len(results)
    if failed:
        missed_gates = float(np.mean([1 - r.gates_passed / r.n_gates for r in failed]))
    else:
        missed_gates = 0.0
    return -mean_lap_time - FAILURE_PENALTY * failure_share - MISSED_GATES_PENALTY * missed_gates
