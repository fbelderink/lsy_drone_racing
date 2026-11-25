import pytest

from lsy_drone_racing.tuning.reward import EpisodeResult, reward


def finished(seed: int, lap_time: float) -> EpisodeResult:
    return EpisodeResult(seed=seed, finished=True, gates_passed=4, n_gates=4, lap_time=lap_time)


def failed(seed: int, gates_passed: int) -> EpisodeResult:
    return EpisodeResult(
        seed=seed, finished=False, gates_passed=gates_passed, n_gates=4, lap_time=3.0
    )


@pytest.mark.unit
def test_all_finished_is_minus_mean_lap_time():
    assert reward([finished(0, 6.0), finished(1, 8.0)]) == pytest.approx(-7.0)


@pytest.mark.unit
def test_failures_add_share_and_missed_gates_penalty():
    # Lap time 6 s, 1 of 4 seeds failed with half the gates missed
    results = [finished(0, 6.0), finished(1, 6.0), finished(2, 6.0), failed(3, gates_passed=2)]
    assert reward(results) == pytest.approx(-6.0 - 100.0 * 0.25 - 5.0 * 0.5)


@pytest.mark.unit
def test_nothing_finished_counts_the_episode_limit_as_lap_time():
    results = [failed(0, gates_passed=0), failed(1, gates_passed=2)]
    assert reward(results) == pytest.approx(-30.0 - 100.0 - 5.0 * 0.75)


@pytest.mark.unit
def test_late_crash_scores_higher_than_early_crash():
    assert reward([failed(0, gates_passed=3)]) > reward([failed(0, gates_passed=1)])


@pytest.mark.unit
def test_one_failure_outweighs_any_speed_gain():
    safe_slow = [finished(seed, 9.0) for seed in range(10)]
    fast_one_failure = [finished(seed, 4.0) for seed in range(9)] + [failed(9, gates_passed=3)]
    assert reward(safe_slow) > reward(fast_one_failure)
