"""What happens when a gate's true position is revealed: speed, distance, needed correction, outcome.

Run as:

    $ python benchmarks/mppi/gate_reveal.py --names=default,sb_lat7_vg30,sb_fast3 --n_seeds=100

A gate's true position is only observed within `sensor_range` (0.7 m). Until then the planner flies
through the nominal gate. At the reveal, the planner moves the gate crossing of the path into the
true opening. This logs, for every reveal:

- the speed and the distance to the gate at the reveal,
- the correction the path needs at the gate (how far the old crossing point lies outside the
  safe opening of the true gate, in the gate plane, split into sideways and vertical),
- whether the drone then passes that gate, or how the episode fails there.

From this the speed at which a given correction can still be flown follows, which is what the
gate speed cap has to respect.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "benchmarks/mppi/results/gate_reveal"


def episode(args: tuple[str, int, int]) -> list[dict]:
    """Run one episode and return one record per gate reveal."""
    name, seed, rng = args
    import gymnasium  # noqa: I001
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
    from variants import VARIANTS

    from lsy_drone_racing.utils import load_config, load_controller

    from bench_mppi import classify

    cfg = load_config(ROOT / "config/level2.toml")
    cfg.sim.render = False
    cfg.controller.mppi = {**VARIANTS[name], "rng_seed": rng}
    cls = load_controller(ROOT / "lsy_drone_racing/control/attitude_mppi.py")
    env = JaxToNumpy(
        gymnasium.make(
            cfg.env.id,
            freq=cfg.env.freq,
            sim_config=cfg.sim,
            sensor_range=cfg.env.sensor_range,
            control_mode=cfg.env.control_mode,
            track=cfg.env.track,
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=seed,
        )
    )
    obs, info = env.reset()
    data = env.unwrapped.data
    gates_true = np.asarray(data.gates_pos[0])
    obstacles_true = np.asarray(data.obstacles_pos[0])
    ctrl = cls(obs, info, cfg)
    planner = ctrl._mlc._planner
    seen = obs["gates_visited"].copy()
    n_total = len(cfg.env.track.gate_order)
    reveals, step, outcome = [], 0, None
    last_pos = obs["pos"]
    while True:
        new = np.flatnonzero(obs["gates_visited"] & ~seen)
        for g in new:  # the reference is still the one through the nominal gate
            forward, _ = planner.gate_plane_crossings(obs, np.array([g]))
            off = forward["offset"]
            if len(off) == 0:
                continue
            o = off[np.argmin(np.linalg.norm(off, axis=1))]  # (y, z) in the true gate frame
            need = np.maximum(np.abs(o) - planner.gate_clearance_inside, 0.0)
            reveals.append(
                {
                    "gate": int(g),
                    "target": int(obs["n_gates_passed"]),
                    "step": step,
                    "speed": float(np.linalg.norm(obs["vel"])),
                    "dist": float(np.linalg.norm(obs["pos"] - obs["gates_pos"][g])),
                    "offset_y": float(o[0]),
                    "offset_z": float(o[1]),
                    "need_y": float(need[0]),
                    "need_z": float(need[1]),
                }
            )
        seen = obs["gates_visited"].copy()
        action = ctrl.compute_control(obs, info)
        last_pos = obs["pos"] + obs["vel"] / cfg.env.freq
        obs, r, term, trunc, info = env.step(action)
        finished = ctrl.step_callback(action, obs, r, term, trunc, info)
        step += 1
        n = int(obs["n_gates_passed"])
        if n == n_total:
            outcome = "success"
        elif term:
            outcome = classify(np.asarray(last_pos), gates_true, obstacles_true)
        elif trunc:
            outcome = "timeout"
        elif finished:
            outcome = "missed_gate"
        if outcome is not None:
            break
    env.close()
    n_final = int(obs["n_gates_passed"])
    for rec in reveals:
        # the reveal belongs to the gate the drone was heading for when it happened
        rec["passed"] = n_final > rec["target"]
        rec["fail_here"] = outcome if (outcome != "success" and n_final == rec["target"]) else None
        rec.update(variant=name, seed=seed, rng=rng, outcome=outcome)
    return reveals


def main(names: str = "default", n_seeds: int = 100, rngs: str = "0", workers: int = 6):
    """Log every gate reveal of the given variants and print a summary."""
    names = names.split(",") if isinstance(names, str) else list(names)
    rngs = [int(r) for r in str(rngs).split(",")] if not isinstance(rngs, tuple) else list(rngs)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    for name in names:
        jobs = [(name, s, r) for r in rngs for s in range(1, n_seeds + 1)]
        with ctx.Pool(workers, maxtasksperchild=5) as pool:
            recs = [x for ep in pool.imap_unordered(episode, jobs) for x in ep]
        (OUT / f"{name}.json").write_text(json.dumps(recs))
        summarize(name, recs)


def summarize(name: str, recs: list[dict]):
    """Failure rate at a gate by the correction needed at its reveal."""
    print(f"\n=== {name}: {len(recs)} reveals")
    need = np.array([np.hypot(r["need_y"], r["need_z"]) for r in recs])
    fail = np.array([r["fail_here"] is not None for r in recs])
    speed = np.array([r["speed"] for r in recs])
    dist = np.array([r["dist"] for r in recs])
    print(f"speed at reveal: median {np.median(speed):.2f} m/s, distance {np.median(dist):.2f} m")
    for lo, hi in ((0, 1e-9), (1e-9, 0.03), (0.03, 0.06), (0.06, 0.1), (0.1, 1)):
        m = (need >= lo) & (need < hi) if lo > 0 else need <= hi
        if m.any():
            print(
                f"  correction {lo:.2f}-{hi:.2f} m: {m.sum():4d} reveals, "
                f"fail at that gate {fail[m].mean():5.1%}, speed {speed[m].mean():.2f} m/s"
            )


def show(names: str = "default"):
    """Print the summary of saved runs."""
    for name in names.split(",") if isinstance(names, str) else names:
        summarize(name, json.loads((OUT / f"{name}.json").read_text()))


if __name__ == "__main__":
    fire.Fire({"run": main, "show": show})
