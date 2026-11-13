"""What limits the lap time: the input limits of the drone or the speed profile of the reference.

Run as:

    $ python benchmarks/mppi/speed_limits.py --seeds=1,2,3,4,5 --variant=short_best

Records closed-loop episodes and prints:

- the drone's thrust-to-weight ratio and the horizontal acceleration it can hold at the tilt limit,
- how the commanded tilt and thrust are distributed and how often they are at their limits,
- the reference speed profile: which constraint (gate cap, lateral acceleration, v_max, the
  acceleration ramps) sets the speed over which share of the path, and the lap time of the profile.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import fire
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))


def profile_breakdown(mlc: Any, ref: Any, obs: dict) -> dict:
    """Split the velocity profile into the constraint that is active at every path point."""
    import numpy.linalg as la

    s = ref.path.x
    kappa = la.norm(ref.path(s, 2), axis=-1)
    v_lat = np.minimum(mlc.v_max, np.sqrt(np.maximum(mlc.a_lat_max / kappa, 1e-6)))
    v = mlc._compute_velocity_profile(ref, obs)
    # the profile without gate caps, to see what the caps cost
    v_gate_max, mlc.v_gate_max = mlc.v_gate_max, mlc.v_max
    v_nocap = mlc._compute_velocity_profile(ref, obs)
    mlc.v_gate_max = v_gate_max
    ds = np.diff(s)
    t = float(np.sum(ds / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)))
    t_nocap = float(np.sum(ds / np.maximum(0.5 * (v_nocap[1:] + v_nocap[:-1]), 1e-3)))
    tol = 1e-3
    active = np.where(
        v >= mlc.v_max - tol,
        "v_max",
        np.where(
            v >= v_lat - tol,
            "a_lat",
            np.where(np.abs(v - v_nocap) > tol, "gate_cap", "accel/brake"),
        ),
    )
    share = {k: float(np.sum(ds[active[:-1] == k]) / np.sum(ds)) for k in np.unique(active)}
    return {"length": float(s[-1]), "t_profile": t, "t_nocap": t_nocap, "share": share}


def record(seed: int, variant: str, config: str = "level2.toml") -> dict:
    """Run one episode and collect commands and the initial reference profile."""
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
    from variants import VARIANTS

    from lsy_drone_racing.utils import load_config, load_controller

    cfg = load_config(ROOT / "config" / config)
    cfg.sim.render = False
    cfg.controller.mppi = dict(VARIANTS[variant])
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
    us, vref, speed, prof = [], [], [], None
    while True:
        action = ctrl.compute_control(obs, info)
        if prof is None:
            ref = ctrl._mlc._planner.plan(obs)
            prof = profile_breakdown(ctrl._mlc, ref, obs)
        us.append(action)
        vref.append(float(np.linalg.norm(ctrl._ref[0][3:6])))
        speed.append(float(np.linalg.norm(obs["vel"])))
        obs, r, term, trunc, info = env.step(action)
        if ctrl.step_callback(action, obs, r, term, trunc, info) or term or trunc:
            break
    # the profile at the end, when every gate has been revealed
    final = profile_breakdown(ctrl._mlc, ctrl._mlc._planner.plan(obs), obs)
    env.close()
    return {
        "us": np.array(us),
        "vref": np.array(vref),
        "speed": np.array(speed),
        "prof": prof,
        "final": final,
        "gates": int(obs["n_gates_passed"]),
        "time": len(us) / cfg.env.freq,
        "params": ctrl._params,
        "hover": ctrl._hover_thrust,
    }


def main(seeds: str | tuple = "1,2,3,4,5", variant: str = "short_best"):
    """Print the input usage and the speed profile breakdown."""
    seeds = [int(s) for s in str(seeds).strip("()").split(",")]
    runs = [record(s, variant) for s in seeds]
    p = runs[0]["params"]
    g = -float(np.asarray(p["gravity_vec"])[-1])
    f_max = float(p["acc_coef"] + p["cmd_f_coef"] * 4 * p["thrust_max"])
    twr = f_max / (float(p["mass"]) * g)
    print(
        f"thrust-to-weight {twr:.2f} (hover command {runs[0]['hover']:.3f}, "
        f"max {4 * p['thrust_max']:.3f})"
    )
    for tilt in (1.0, 1.2, np.arccos(1 / twr) if twr > 1 else 0.0):
        # horizontal acceleration at this tilt with the thrust that holds the altitude, if it fits
        need = 1 / np.cos(tilt)
        a = g * np.tan(tilt) if need <= twr else g * np.sqrt(max(twr**2 - 1, 0))
        print(
            f"  tilt {np.degrees(tilt):5.1f} deg: thrust needed {need:.2f} g, "
            f"max level a_h {a:5.1f} m/s^2"
        )
    us = np.concatenate([r["us"] for r in runs])
    tilt = np.arccos(np.cos(us[:, 0]) * np.cos(us[:, 1]))
    thr = us[:, 3] / (4 * p["thrust_max"])
    print(f"\n{len(us)} steps, {variant}")
    for q in (50, 90, 99, 100):
        print(
            f"  p{q:<3} |roll| {np.percentile(np.abs(us[:, 0]), q):.2f}  |pitch| "
            f"{np.percentile(np.abs(us[:, 1]), q):.2f}  tilt {np.percentile(tilt, q):.2f} rad  "
            f"thrust {np.percentile(thr, q):.2f} of max"
        )
    at_min = thr <= 0.02 + p["thrust_min"] / p["thrust_max"]
    print(
        f"  at limit: roll/pitch >= 0.95 rad {np.mean(np.max(np.abs(us[:, :2]), 1) >= 0.95):.1%}, "
        f"thrust >= 0.98 max {np.mean(thr >= 0.98):.1%}, "
        f"thrust <= min+2% {np.mean(at_min):.1%}"
    )
    print("\nreference speed profile (first plan, nominal gates / final plan, all gates revealed)")
    for r in runs:
        a, b = r["prof"], r["final"]
        share = " ".join(f"{k} {v:.0%}" for k, v in sorted(a["share"].items()))
        print(
            f"  len {a['length']:.1f} m  t_profile {a['t_profile']:.2f}s (no gate caps "
            f"{a['t_nocap']:.2f}s, final {b['t_profile']:.2f}s)  flown {r['time']:.2f}s "
            f"gates {r['gates']}  v_mean {r['speed'].mean():.2f} "
            f"vref_mean {r['vref'].mean():.2f}  [{share}]"
        )


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
