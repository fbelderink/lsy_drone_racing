"""Why the drone overshoots turns on fast speed profiles: plan, model or attitude tracking.

Run as:

    $ python benchmarks/mppi/overshoot.py --seed=4 --variant=sb_lat9_vg20

Replays one episode and splits the error to the reference into three parts at every step:

- plan error: the MPPI mean plan (rolled out with the planning model) vs. the reference window.
  Large means the optimizer chooses to deviate (cost or sampling problem).
- prediction error: the mean plan vs. the positions the drone actually reaches. Large means the
  planning model is wrong (model problem).
- attitude lag: commanded roll/pitch vs. the drone's actual roll/pitch.

Prints these over time around the largest tracking error and as episode averages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))


def replay(seed: int, variant: str, rng: int = 0, config: str = "level2.toml") -> dict:
    """Run one episode and log plan, reference window, state and command at every step."""
    import gymnasium  # noqa: I001
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
    from variants import VARIANTS

    from lsy_drone_racing.utils import load_config, load_controller  # sets SCIPY_ARRAY_API first

    from scipy.spatial.transform import Rotation as R  # noqa: I001

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
    log = {k: [] for k in ("pos", "vel", "rpy", "u", "ref", "plan", "gates")}
    while True:
        action = ctrl.compute_control(obs, info)
        log["pos"].append(obs["pos"].copy())
        log["vel"].append(obs["vel"].copy())
        log["rpy"].append(R.from_quat(obs["quat"]).as_euler("xyz"))
        log["u"].append(np.asarray(action).copy())
        log["ref"].append(np.asarray(ctrl._ref)[:, :3].copy())  # (H + 1, 3) at the knots
        log["plan"].append(np.asarray(ctrl._vis[2]).copy())  # (H + 1, 3) mean plan
        log["gates"].append(int(obs["n_gates_passed"]))
        obs, r, term, trunc, info = env.step(action)
        if ctrl.step_callback(action, obs, r, term, trunc, info) or term or trunc:
            break
    env.close()
    out = {k: np.array(v) for k, v in log.items()}
    out["knots"] = np.asarray(ctrl._knot_idx)
    return out


def errors(d: dict) -> dict:
    """Plan, prediction and attitude errors per step and look-ahead knot."""
    n, knots = len(d["pos"]), d["knots"]
    plan_err = np.linalg.norm(d["plan"] - d["ref"], axis=-1)  # (n, H + 1)
    pred_err = np.full(plan_err.shape, np.nan)
    for j, kj in enumerate(knots):
        valid = np.arange(n) + kj < n
        pred_err[valid, j] = np.linalg.norm(
            d["plan"][valid, j] - d["pos"][np.arange(n)[valid] + kj], axis=-1
        )
    track_err = np.linalg.norm(d["pos"] - d["ref"][:, 0], axis=-1)
    att_lag = np.abs(d["u"][:, :2] - d["rpy"][:, :2]).max(axis=1)
    return {"plan": plan_err, "pred": pred_err, "track": track_err, "att": att_lag}


def main(seed: int = 4, variant: str = "sb_lat9_vg20", compare: str = "default", rng: int = 0):
    """Print where the error to the reference comes from, for a variant and a reference variant."""
    for name in (variant, compare):
        d = replay(seed, name, rng)
        e = errors(d)
        knots = d["knots"]
        js = [j for j, k in enumerate(knots) if k in (10, 20, 30)]
        t = [knots[j] / 50 for j in js]
        print(f"\n=== {name}, seed {seed}: {len(d['pos'])} steps, gates {d['gates'][-1]}")
        print(
            "episode mean:  track {:.3f} m | ".format(np.mean(e["track"]))
            + " ".join(
                f"plan@{tt:.1f}s {np.nanmean(e['plan'][:, j]):.3f} pred@{tt:.1f}s "
                f"{np.nanmean(e['pred'][:, j]):.3f}"
                for j, tt in zip(js, t)
            )
            + f" | att lag {np.mean(e['att']):.3f} rad"
        )
        k0 = int(np.argmax(e["track"]))
        print(f"around the largest tracking error (step {k0}, {e['track'][k0]:.2f} m):")
        print("  k   track  |v|   plan@0.2 plan@0.4 plan@0.6  pred@0.2 pred@0.4 pred@0.6  att   g")
        for k in range(max(0, k0 - 36), min(len(d["pos"]), k0 + 4), 3):
            print(
                f"{k:4d}  {e['track'][k]:.2f}  {np.linalg.norm(d['vel'][k]):.2f}   "
                + "  ".join(f"{e['plan'][k, j]:6.2f}" for j in js)
                + "   "
                + "  ".join(f"{e['pred'][k, j]:6.2f}" for j in js)
                + f"   {e['att'][k]:.2f}  {d['gates'][k]}"
            )


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
