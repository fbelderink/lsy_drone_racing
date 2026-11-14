"""Dynamic feasibility of the mid-level speed profile.

Run as:

    $ python benchmarks/mppi/profile_feasibility.py --seed=4

Plans the path from the first observation of a seed, computes the speed profile for several sets of
mid-level limits and samples the resulting reference in time. A quadrotor must produce the thrust
`a + g` with its body axis, so for every profile it prints:

- the profile lap time,
- how often the reference needs a vertical thrust below 0.2 g (accelerating downwards almost as
  fast as or faster than gravity), more than 1.8 g in total (the drone's maximum) or a tilt above
  56 degrees,
- how fast the required thrust direction turns. The drone reaches at most ~500 deg/s in closed loop.
"""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).parents[2]
G = 9.81
F16 = {"thrust_max": 1.6 * G, "thrust_min": 0.3 * G, "tilt_max": np.radians(55)}
CASES = [  # (a_lat, a_long, a_brake, extra MidLevelController arguments)
    (5, 8, 7, {}),
    (7, 10, 10, {}),
    (7, 10, 10, {"smooth_time": 0.25}),
    (9, 10, 10, {}),
    (9, 10, 10, {"smooth_time": 0.15}),
    (12, 12, 12, {}),
    (20, 10, 10, F16),
    (20, 10, 10, {**F16, "smooth_time": 0.15}),
    (20, 10, 10, {**F16, "smooth_time": 0.25}),
]


def demand(p: np.ndarray, dt: float) -> tuple[np.ndarray, ...]:
    """Required tilt [deg], thrust-axis rate [deg/s], specific thrust and its z part [m/s^2]."""
    vel = np.gradient(p, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    k = np.ones(10) / 10  # average over 20 ms, one control step, against sampling noise
    acc = np.stack([np.convolve(acc[:, i], k, "same") for i in range(3)], axis=1)
    f = acc + np.array([0.0, 0.0, G])
    norm = np.linalg.norm(f, axis=1)
    n = f / norm[:, None]
    tilt = np.degrees(np.arccos(np.clip(n[:, 2], -1, 1)))
    rate = np.degrees(np.arccos(np.clip(np.sum(n[1:] * n[:-1], axis=1), -1, 1))) / dt
    return tilt, rate, norm, f[:, 2]


def main(seed: int = 4, config: str = "level2.toml"):
    """Print the feasibility of the speed profile for several mid-level limits."""
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    from lsy_drone_racing.control.mid_level.mid_level_controller import MidLevelController
    from lsy_drone_racing.control.planner.splines import SplinePlanner
    from lsy_drone_racing.utils import load_config

    cfg = load_config(ROOT / "config" / config)
    cfg.sim.render = False
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
    obs, _ = env.reset()
    env.close()
    planner = SplinePlanner(obs, cfg)
    ref = planner.plan(obs)
    s = ref.path.x
    su = np.linspace(0, s[-1], 4000)
    dt = 0.002
    print(f"seed {seed}: path {s[-1]:.1f} m")
    print(
        f"{'a_lat':>5} {'a_lo':>4} {'a_br':>4} {'extra':<38} {'lap':>6} {'z<0.2g':>6} {'>1.8g':>6} "
        f"{'>56deg':>6} {'rate p95':>9} {'>500/s':>7}"
    )
    for a_lat, a_long, a_brake, kw in CASES:
        mlc = MidLevelController(
            planner, 6.0, a_long, a_lat, a_brake, v_gate_max=2.0, gate_window=1.0,
            gate_plateau=0.9, **kw,
        )  # fmt: skip
        v = np.interp(su, s, mlc._compute_velocity_profile(ref, obs))
        t = np.concatenate(([0], np.cumsum(np.diff(su) / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3))))
        tt = np.arange(0, t[-1], dt)
        tilt, rate, norm, f_z = demand(ref.path(np.interp(tt, t, su)), dt)
        extra = ", ".join(f"{k}={float(x):.3g}" for k, x in kw.items())
        print(
            f"{a_lat:>5} {a_long:>4} {a_brake:>4} {extra:<38} {t[-1]:5.2f}s "
            f"{np.mean(f_z < 0.2 * G):6.1%} {np.mean(norm > 1.8 * G):6.1%} "
            f"{np.mean(tilt > 56):6.1%} "
            f"{np.percentile(rate, 95):7.0f}/s {np.mean(rate > 500):7.1%}"
        )


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
