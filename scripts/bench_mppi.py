"""Benchmark controller variants over many seeds in parallel processes.

Run as:

    $ python scripts/bench_mppi.py --names=baseline,beta06 --n_seeds=50

Variants are defined in `benchmarks/mppi/variants.py` as overrides of the MPPI parameters (see
`DEFAULT_PARAMS` in `attitude_mppi.py`). Each variant is evaluated on the seeds 1 to `n_seeds` of
the given config, so all variants run on identical tracks and disturbances. Every episode records
the outcome (success, gate, obstacle, out of bounds, ground, timeout, missed gate), the number of
gates passed, the lap time, and the wall time per MPPI update.

Results are written to `benchmarks/mppi/results/<config stem>/<variant>.json` and a summary table
is printed. `--compare=baseline` prints paired differences against another variant's results.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import fire
import numpy as np

ROOT = Path(__file__).parents[1]
# Episodes per worker process before it is replaced. A worker's memory grows with every episode
# (2.5-3 GB after 10 with the long horizon), so recycling keeps the total bounded
TASKS_PER_WORKER = 5
RESULTS = ROOT / "benchmarks" / "mppi" / "results"
logger = logging.getLogger("bench_mppi")


def _worker_init(core_queue: Any):
    """Pin this worker to one physical core (and its SMT sibling) before JAX starts."""
    cores = core_queue.get()
    # Workers are recycled (see `TASKS_PER_WORKER`), so a worker hands its core back on exit
    atexit.register(core_queue.put, cores)
    try:
        os.sched_setaffinity(0, cores)
    except (AttributeError, OSError):
        pass
    # Exit if the benchmark process dies, instead of running on as an orphan and holding memory
    parent = os.getppid()

    def watchdog():
        while True:
            time.sleep(5)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=watchdog, daemon=True).start()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)


def classify(pos: np.ndarray, gates_pos: np.ndarray, obstacles_pos: np.ndarray) -> str:
    """Name the most likely cause of a crash at `pos` from the true object positions."""
    if pos[2] <= 0.02:
        return "ground"
    if np.any(np.abs(pos[:2]) > 2.95) or pos[2] > 2.45:
        return "out_of_bounds"
    d_obstacle = np.min(np.linalg.norm(obstacles_pos[:, :2] - pos[:2], axis=-1))
    d_gate = np.min(np.linalg.norm(gates_pos - pos, axis=-1))
    # Obstacles are thin poles, gate frames extend up to 0.36 m from the gate's center
    if d_obstacle < 0.12 and d_obstacle < d_gate - 0.2:
        return "obstacle"
    return "gate" if d_gate < 0.7 else "other"


MIN_AVAILABLE_GB = 4.0  # a worker waits before an episode while less memory is available


def _available_gb() -> float:
    """Available memory in GB from /proc/meminfo, or infinity where it does not exist."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("inf")


def run_episode(config_name: str, controller: str, params: dict, seed: int) -> dict:
    """Run one headless episode and return its result record."""
    while _available_gb() < MIN_AVAILABLE_GB:  # back off instead of driving the machine into OOM
        time.sleep(5)
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    from lsy_drone_racing.utils import load_config, load_controller

    config = load_config(ROOT / "config" / config_name)
    config.sim.render = False
    config.controller.mppi = params
    controller_cls = load_controller(Path(controller))

    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=seed,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset()
    t0 = time.perf_counter()
    ctrl = controller_cls(obs, info, config)
    init_s = time.perf_counter() - t0
    data = env.unwrapped.data
    gates_pos = np.asarray(data.gates_pos[0])
    obstacles_pos = np.asarray(data.obstacles_pos[0])
    n_total = len(config.env.track.gate_order)

    steps, ess, speed = 0, [], []
    last_pos = obs["pos"]
    outcome = None
    while True:
        action = ctrl.compute_control(obs, info)
        ess.append(getattr(ctrl, "_ess", np.nan))
        last_pos = obs["pos"] + obs["vel"] / config.env.freq
        speed.append(float(np.linalg.norm(obs["vel"])))
        obs, reward, terminated, truncated, info = env.step(action)
        finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        steps += 1
        n_passed = int(obs["n_gates_passed"])
        if n_passed == n_total:
            outcome = "success"
        elif terminated:
            outcome = classify(np.asarray(last_pos), gates_pos, obstacles_pos)
        elif truncated:
            outcome = "timeout"
        elif finished:
            outcome = "missed_gate"  # the reference ended before all gates were passed
        if outcome is not None:
            break
    env.close()
    step_ms = np.asarray(getattr(ctrl, "_step_ms", [np.nan]))
    return {
        "seed": seed,
        "rng": int(params.get("rng_seed", 0)),
        "outcome": outcome,
        "gates": int(obs["n_gates_passed"]),
        "time": steps / config.env.freq,
        "ms_mean": float(np.mean(step_ms)),
        "ms_p95": float(np.percentile(step_ms, 95)),
        "ess_median": float(np.nanmedian(ess)),
        "v_max": float(np.max(speed)),
        "v_mean": float(np.mean(speed)),
        "init_s": init_s,
    }


def summarize(name: str, records: list[dict]) -> dict:
    """Aggregate the episode records of one variant."""
    n = len(records)
    ok = [r for r in records if r["outcome"] == "success"]
    outcomes = {}
    for r in records:
        if r["outcome"] != "success":
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    times = np.array([r["time"] for r in ok])
    return {
        "name": name,
        "n": n,
        "success": len(ok),
        "rate": len(ok) / n,
        "time_mean": float(times.mean()) if len(ok) else float("nan"),
        "time_median": float(np.median(times)) if len(ok) else float("nan"),
        "time_std": float(times.std()) if len(ok) else float("nan"),
        "ms_mean": float(np.mean([r["ms_mean"] for r in records])),
        "v_max": float(np.mean([r["v_max"] for r in records])),
        "failures": outcomes,
        "failed_seeds": sorted(
            (r["seed"], r.get("rng", 0), r["outcome"][:4], r["gates"])
            for r in records
            if r["outcome"] != "success"
        ),
    }


def fmt_row(s: dict) -> str:
    """Format a summary as one table row."""
    fails = " ".join(f"{k}:{v}" for k, v in sorted(s["failures"].items())) or "-"
    return (
        f"{s['name']:<28} {s['success']:>3}/{s['n']:<3} {s['rate']:5.1%} "
        f"{s['time_mean']:6.3f} {s['time_median']:6.3f} {s['time_std']:5.3f} "
        f"{s['ms_mean']:6.1f} {s['v_max']:5.2f}  {fails}"
    )


HEADER = (
    f"{'variant':<28} {'succ':>7} {'rate':>5} {'t_mean':>6} {'t_med':>6} {'t_std':>5} "
    f"{'ms':>6} {'vmax':>5}  failures"
)


def load_results(config: str, name: str) -> list[dict] | None:
    """Load the stored records of a variant, or None if it was not run."""
    path = RESULTS / Path(config).stem / f"{name}.json"
    return json.loads(path.read_text())["records"] if path.exists() else None


def paired(base: list[dict], other: list[dict]) -> str:
    """Compare two variants seed by seed."""
    b = {(r["seed"], r.get("rng", 0)): r for r in base}
    o = {(r["seed"], r.get("rng", 0)): r for r in other}
    seeds = sorted(set(b) & set(o))
    both = [s for s in seeds if b[s]["outcome"] == "success" and o[s]["outcome"] == "success"]
    gained = [s for s in seeds if b[s]["outcome"] != "success" and o[s]["outcome"] == "success"]
    lost = [s for s in seeds if b[s]["outcome"] == "success" and o[s]["outcome"] != "success"]
    dt = np.array([o[s]["time"] - b[s]["time"] for s in both])
    if len(dt):
        se = dt.std(ddof=1) / np.sqrt(len(dt)) if len(dt) > 1 else float("nan")
        dts = f"dt {dt.mean():+.3f}s ± {se:.3f} (paired, n={len(dt)})"
    else:
        dts = "dt n/a"
    return f"{dts}, gained {len(gained)} {gained}, lost {len(lost)} {lost}"


def main(
    names: str | tuple = "baseline",
    n_seeds: int = 50,
    seed_start: int = 1,
    config: str = "level2.toml",
    controller: str = "attitude_mppi.py",
    workers: int = 8,
    compare: str | None = "baseline",
    rerun: bool = True,
    show: bool = False,
    rngs: str | int | tuple = 0,
    core_offset: int = 0,
):
    """Benchmark the given variants.

    Args:
        names: Comma-separated variant names from `benchmarks/mppi/variants.py`.
        n_seeds: Number of seeds per variant.
        seed_start: First seed.
        config: Config file in `config/`.
        controller: Controller file in `lsy_drone_racing/control/`.
        workers: Number of parallel processes, each pinned to one physical core.
        compare: Variant to compare against seed by seed, or None.
        rerun: If False, variants with stored results are not run again.
        show: Only print the stored results of the given variants.
        rngs: Comma-separated seeds of the MPPI sampling noise. Every (seed, rng) pair is one
            episode, so several rngs average over the sampling noise of the controller.
        core_offset: First physical core to pin workers to, to run two benchmarks side by side.
    """
    sys.path.insert(0, str(ROOT / "benchmarks" / "mppi"))
    from variants import VARIANTS

    names = names.split(",") if isinstance(names, str) else list(names)
    out_dir = RESULTS / Path(config).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(seed_start, seed_start + n_seeds))
    rngs = [int(r) for r in str(rngs).strip("()").split(",") if r.strip() != ""]

    todo = [] if show else [n for n in names if rerun or load_results(config, n) is None]
    if todo:
        # Snapshot the controller, so that editing it during a long run does not change the results
        snap_dir = Path(tempfile.mkdtemp(prefix="bench_mppi_"))
        snapshot = snap_dir / controller
        shutil.copy(ROOT / "lsy_drone_racing/control" / controller, snapshot)
        controller = str(snapshot)
        n_cpu = os.cpu_count() // 2
        ctx = get_context("spawn")
        queue = ctx.Queue()
        for i in range(workers):
            core = (core_offset + i) % n_cpu
            queue.put({core, core + n_cpu})
        with ProcessPoolExecutor(
            workers, ctx, _worker_init, (queue,), max_tasks_per_child=TASKS_PER_WORKER
        ) as pool:
            for name in todo:
                t0 = time.time()
                futures = [
                    pool.submit(
                        run_episode, config, controller, {**VARIANTS[name], "rng_seed": r}, s
                    )
                    for r in rngs
                    for s in seeds
                ]
                records = [f.result() for f in futures]
                out = {"variant": name, "params": VARIANTS[name], "config": config}
                out["records"] = records
                (out_dir / f"{name}.json").write_text(json.dumps(out, separators=(",", ":")))
                print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)

    print(HEADER)
    for name in names:
        rec = load_results(config, name)
        if rec is None:
            print(f"{name:<28} (no results)")
            continue
        print(fmt_row(summarize(name, rec)))
    if compare is not None and (base := load_results(config, compare)) is not None:
        for name in names:
            if name != compare and (rec := load_results(config, name)) is not None:
                print(f"  {name} vs {compare}: {paired(base, rec)}")
    for name in names:
        if (rec := load_results(config, name)) is not None:
            fails = summarize(name, rec)["failed_seeds"]
            print(f"  {name} failed (seed, rng, cause, gates): {fails}")


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
