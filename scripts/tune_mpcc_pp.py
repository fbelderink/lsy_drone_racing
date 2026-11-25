"""Tune the MPCC++ controller with TuRBO-1 on seeded level-2 episodes.

Every candidate runs on the same seeds in parallel worker processes; its reward combines the share
of finished episodes and their mean lap time (see `lsy_drone_racing/tuning/reward.py`). Results are
appended to `<out>/results.jsonl` after every round, from which `--resume` continues.

Run in the `tune` pixi environment:

    $ python scripts/tune_mpcc_pp.py tune --out tuning_runs/first --workers 20
    $ python scripts/tune_mpcc_pp.py tune --out tuning_runs/first --resume
    $ python scripts/tune_mpcc_pp.py check --out tuning_runs/first --top 5
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import numpy as np
import torch

from lsy_drone_racing.tuning.episode import run_episode
from lsy_drone_racing.tuning.reward import EpisodeResult, reward
from lsy_drone_racing.tuning.search_space import (
    SEARCH_SPACE,
    default_values,
    unit_to_values,
    values_to_overrides,
    values_to_unit,
)
from lsy_drone_racing.tuning.turbo import Evaluation, TurboState, next_batch, update_state

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

CONTROLLER = "attitude_mppi_mpcc_pp.py"
# The first nine seeds finish with the untuned controller, seed 29 hits a pole
TUNING_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 29)
CHECK_SEEDS = tuple(range(40, 60))


def evaluate(
    pool: ProcessPoolExecutor, config: str, seeds: Sequence[int], candidates: list[dict]
) -> list[list[EpisodeResult]]:
    """Run every candidate on every seed in parallel.

    Args:
        pool: Worker processes.
        config: Config file name in `config/`.
        seeds: Seeds of the episodes.
        candidates: Controller overrides of every candidate.

    Returns:
        The episode results of every candidate, in the order of seeds.
    """
    futures = [
        [pool.submit(run_episode, config, CONTROLLER, seed, overrides) for seed in seeds]
        for overrides in candidates
    ]
    return [[future.result() for future in candidate] for candidate in futures]


def summary(results: Sequence[EpisodeResult]) -> str:
    """Share of finished episodes and their mean lap time, e.g. "9/10 finished, 7.32 s"."""
    lap_times = [result.lap_time for result in results if result.finished]
    mean_lap_time = f"{np.mean(lap_times):.2f} s" if lap_times else "-"
    return f"{len(lap_times)}/{len(results)} finished, {mean_lap_time}"


def read_records(path: Path) -> list[dict]:
    """Read the records of `results.jsonl`, one per evaluated candidate."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_evaluation(record: dict) -> Evaluation:
    """The part of a record that TuRBO needs."""
    state = None if record["state"] is None else TurboState(**record["state"])
    return Evaluation(record["run"], tuple(record["x"]), record["reward"], state)


def warm_start_record(record: dict, source: str) -> dict:
    """Place a record of another search, possibly over fewer variables, into this search space.

    Variables that the other search did not tune had the controller's defaults, so they get those.
    Its trust region continues with this search's dimension.

    Args:
        record: Record of the other search's `results.jsonl`.
        source: Path of that file, stored in the record.

    Returns:
        The record in this search space.
    """
    values = default_values() | record["values"]
    state = record["state"]
    if state is not None:
        state = state | {"dim": len(SEARCH_SPACE)}
    x = values_to_unit(values).tolist()
    return record | {"x": x, "values": values, "state": state, "warm_start": source}


def tune(
    out: str | None = None,
    config: str = "level2.toml",
    seeds: Sequence[int] = TUNING_SEEDS,
    n_candidates: int = 200,
    batch: int = 2,
    n_init: int = 20,
    workers: int = 20,
    resume: bool = False,
    warm_start: str | None = None,
):
    """Search the parameters with TuRBO-1.

    A warm start copies the evaluations of another search into this one and continues it: its
    initial design and trust region carry over, and its evaluations count towards n_candidates.

    Args:
        out: Output directory, `tuning_runs/<timestamp>` if None.
        config: Config file name in `config/`.
        seeds: Seeds of every candidate's episodes.
        n_candidates: Total number of candidates, including those of earlier sessions.
        batch: Candidates per round.
        n_init: Size of the initial design of every run of TuRBO-1. The first one starts with the
            controller's defaults.
        workers: Number of parallel episodes.
        resume: Continue from `<out>/results.jsonl`.
        warm_start: `results.jsonl` of another search to start from, for a new search only.
    """
    out_dir = Path(out or f"tuning_runs/{datetime.now():%Y%m%d_%H%M%S}")
    log_path = out_dir / "results.jsonl"
    if resume:
        records = read_records(log_path)
        assert all(record["seeds"] == list(seeds) for record in records), "Seeds changed"
    else:
        assert not log_path.exists(), f"{log_path} exists, pass --resume to continue"
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        if warm_start is not None:
            records = [warm_start_record(r, warm_start) for r in read_records(Path(warm_start))]
            assert all(record["seeds"] == list(seeds) for record in records), "Seeds differ"
            with open(log_path, "w") as f:
                f.writelines(json.dumps(record) + "\n" for record in records)
            logger.info(f"Warm start from {len(records)} evaluations of {warm_start}")
    history = [to_evaluation(record) for record in records]
    x_default = torch.tensor(values_to_unit(default_values()), dtype=torch.float64)

    # Spawned workers: forked ones would inherit the parent's JAX and torch thread pools
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        while len(history) < n_candidates:
            t_start = time.perf_counter()
            dim = len(SEARCH_SPACE)
            X_next, run, state = next_batch(history, dim, n_init, batch, x_default)
            X_next = X_next[: n_candidates - len(history)]
            values = [unit_to_values(x.numpy()) for x in X_next]
            results = evaluate(pool, config, seeds, [values_to_overrides(v) for v in values])
            rewards = [reward(candidate) for candidate in results]
            if state is not None:
                state = update_state(state, rewards)

            with open(log_path, "a") as f:
                for x, value, candidate, candidate_reward in zip(
                    X_next, values, results, rewards, strict=True
                ):
                    record = {
                        "run": run,
                        "x": x.tolist(),
                        "values": value,
                        "reward": candidate_reward,
                        "seeds": list(seeds),
                        "results": [asdict(result) for result in candidate],
                        "state": None if state is None else asdict(state),
                    }
                    f.write(json.dumps(record) + "\n")
                    history.append(to_evaluation(record))

            best = max(history, key=lambda evaluation: evaluation.reward)
            round_summary = "; ".join(
                f"{r:.2f} ({summary(c)})" for r, c in zip(rewards, results, strict=True)
            )
            length = "design" if state is None else f"length {state.length:.3f}"
            logger.info(
                f"{len(history)}/{n_candidates} run {run} {length}: {round_summary}. "
                f"Best {best.reward:.2f}. {time.perf_counter() - t_start:.0f} s"
            )


def check(
    out: str,
    config: str = "level2.toml",
    seeds: Sequence[int] = CHECK_SEEDS,
    top: int = 5,
    workers: int = 20,
):
    """Run the best candidates and the controller's defaults on seeds not used for tuning.

    Prints the results and writes them to `<out>/check.jsonl`.

    Args:
        out: Output directory of `tune`.
        config: Config file name in `config/`.
        seeds: Seeds of the episodes.
        top: Number of best candidates to check.
        workers: Number of parallel episodes.
    """
    out_dir = Path(out)
    records = sorted(read_records(out_dir / "results.jsonl"), key=lambda r: -r["reward"])
    candidates = [("defaults", default_values())]
    # Variables that the search did not tune had the controller's defaults
    candidates += [
        (f"tuned #{rank + 1}", default_values() | r["values"])
        for rank, r in enumerate(records[:top])
    ]

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        overrides = [values_to_overrides(values) for _, values in candidates]
        results = evaluate(pool, config, seeds, overrides)

    with open(out_dir / "check.jsonl", "w") as f:
        for (name, values), candidate in zip(candidates, results, strict=True):
            record = {
                "name": name,
                "values": values,
                "reward": reward(candidate),
                "seeds": list(seeds),
                "results": [asdict(result) for result in candidate],
            }
            f.write(json.dumps(record) + "\n")
            logger.info(f"{name}: reward {reward(candidate):.2f}, {summary(candidate)}")
            logger.info("    " + ", ".join(f"{k}={v:.4g}" for k, v in values.items()))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    fire.Fire({"tune": tune, "check": check})
