"""Replay one seed and print the last steps before the episode ends, for diagnosing failures.

Run as:

    $ python benchmarks/mppi/diagnose.py --seed=7 --variant=baseline

Prints per step: drone position and speed, the reference point and speed, the tracking error, the
ESS, the gates passed, and the distances to the nearest true gate center and obstacle pole.
Optionally saves the trajectory (`--save=path.npz`) for plotting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))


def main(
    seed: int = 7,
    variant: str = "baseline",
    config: str = "level2.toml",
    last: int = 40,
    every: int = 2,
    save: str | None = None,
    rng: int = 0,
):
    """Replay a seed and print the end of the episode."""
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
    from variants import VARIANTS

    from lsy_drone_racing.utils import load_config, load_controller

    cfg = load_config(ROOT / "config" / config)
    cfg.sim.render = False
    cfg.controller.mppi = {**VARIANTS[variant], "rng_seed": rng}
    cls = load_controller(ROOT / "lsy_drone_racing/control/attitude_mppi.py")
    env = gymnasium.make(
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
    env = JaxToNumpy(env)
    obs, info = env.reset()
    ctrl = cls(obs, info, cfg)
    gates = np.asarray(env.unwrapped.data.gates_pos[0])
    obstacles = np.asarray(env.unwrapped.data.obstacles_pos[0])
    print("true gates\n", gates.round(3), "\ntrue obstacles\n", obstacles.round(3))
    print("nominal gates\n", np.asarray(obs["gates_pos"]).round(3))
    log = []
    while True:
        action = ctrl.compute_control(obs, info)
        ref = ctrl._ref[0]
        log.append(
            dict(
                pos=obs["pos"].copy(),
                vel=obs["vel"].copy(),
                ref=np.asarray(ref[:3]),
                vref=float(np.linalg.norm(ref[3:6])),
                ess=ctrl._ess,
                gates=int(obs["n_gates_passed"]),
                action=action.copy(),
                mean_path=ctrl._vis[2].copy(),
            )
        )
        obs, r, term, trunc, info = env.step(action)
        done = ctrl.step_callback(action, obs, r, term, trunc, info)
        if term or trunc or done:
            break
    print(f"ended after {len(log)} steps, gates passed {int(obs['n_gates_passed'])}, term={term}")
    print(" k    pos                 |v|   ref                 |vref|  err   ess  g  d_gate d_obs")
    for k in range(max(0, len(log) - last), len(log), every):
        e = log[k]
        d_gate = np.linalg.norm(gates - e["pos"], axis=1)
        d_obs = np.linalg.norm(obstacles[:, :2] - e["pos"][:2], axis=1)
        print(
            f"{k:3d} {np.array2string(e['pos'], precision=2):20s} {np.linalg.norm(e['vel']):4.2f} "
            f"{np.array2string(e['ref'], precision=2):20s} {e['vref']:4.2f}  "
            f"{np.linalg.norm(e['pos'] - e['ref']):4.2f} {e['ess']:5.0f} {e['gates']}  "
            f"{d_gate.min():4.2f}@{d_gate.argmin()} {d_obs.min():4.2f}@{d_obs.argmin()}"
        )
    if save:
        np.savez(
            save,
            pos=np.array([e["pos"] for e in log]),
            ref=np.array([e["ref"] for e in log]),
            gates=gates,
            obstacles=obstacles,
            mean_path=np.array([e["mean_path"] for e in log]),
        )


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
