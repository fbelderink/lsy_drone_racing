"""Open-loop accuracy of the planning model for different time grids and integrators.

Run as:

    $ python benchmarks/mppi/model_accuracy.py --seeds=1,2,3

Records closed-loop episodes of the baseline controller in the environment (`first_principles`
dynamics), then predicts the drone's position from every 5th state with the applied commands, on
each time grid with each integrator. Within a coarse step the model holds the command of the step's
first control period, as the planner does. Prints the mean position error at several prediction
times.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))


def record(seed: int, config: str = "level2.toml") -> tuple[np.ndarray, np.ndarray]:
    """Run the baseline controller and return the planning states and the applied actions."""
    import gymnasium
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    from lsy_drone_racing.utils import load_config, load_controller

    cfg = load_config(ROOT / "config" / config)
    cfg.sim.render = False
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
    xs, us = [], []
    while True:
        xs.append(np.asarray(ctrl._state(obs)))
        action = ctrl.compute_control(obs, info)
        us.append(action)
        obs, r, term, trunc, info = env.step(action)
        if ctrl.step_callback(action, obs, r, term, trunc, info) or term or trunc:
            break
    return np.array(xs), np.array(us), ctrl._params


def main(seeds: str | tuple = "1,2,3", every: int = 5, t_eval: tuple = (0.3, 0.6, 1.0)):
    """Print open-loop position errors per grid and integrator."""
    import jax
    import jax.numpy as jnp

    import lsy_drone_racing.control.attitude_mppi as m

    seeds = [int(s) for s in str(seeds).strip("()").split(",")]
    data = [record(s) for s in seeds]
    params = data[0][2]
    dt = 0.02
    # Oracle: the constant acceleration bias of each episode, the mean gap between the observed
    # velocity change and the model's prediction over one control period
    f = jax.jit(jax.vmap(lambda x, u: m.state_derivative(x, u, params)))
    with_bias = []
    for xs, us, p in data:
        acc_model = np.asarray(f(jnp.asarray(xs[:-1]), jnp.asarray(us[:-1])))[:, 6:9]
        acc_true = (xs[1:, 6:9] - xs[:-1, 6:9]) / dt
        bias = np.mean(acc_true - acc_model, axis=0)
        print(f"oracle bias {np.round(bias, 3)} m/s^2")
        with_bias.append((xs, us, p, bias))
    data = with_bias
    grids = {
        "uniform 0.02": [1] * 50,
        "uniform 0.04": [2] * 25,
        "uniform 0.06": [3] * 17,
        "uniform 0.10": [5] * 10,
        "uniform 0.12": [6] * 9,
        "10x0.02+15x0.06": [1] * 10 + [3] * 15,
        "10x0.02+10x0.08": [1] * 10 + [4] * 10,
    }
    print(f"{'grid':<18} {'integrator':<8} " + " ".join(f"err@{t:.1f}s" for t in t_eval))
    for name, factors in grids.items():
        factors = np.array(factors)
        dts = (factors * dt).astype(np.float32)
        knots = np.concatenate(([0], np.cumsum(factors)))  # control periods at every knot
        disc = m.discretize(params, dts)
        for integrator in ("euler", "exact", "rk4", "exact+bias"):
            integ, use_bias = integrator.split("+")[0], integrator.endswith("bias")
            fn = jax.jit(
                jax.vmap(
                    lambda x0, U, b, integ=integ: m.rollout(
                        x0, U, params, jnp.asarray(dts), disc, integ, b
                    ),
                    in_axes=(0, 0, None),
                )
            )
            errs = {t: [] for t in t_eval}
            for xs, us, _, bias in data:
                b = jnp.asarray(bias if use_bias else np.zeros(3), dtype=jnp.float32)
                starts = np.arange(0, len(xs) - knots[-1] - 1, every)
                if len(starts) == 0:
                    continue
                U = np.stack([us[s + knots[:-1]] for s in starts])  # command at each step's start
                X = np.asarray(fn(jnp.asarray(xs[starts]), jnp.asarray(U), b))
                for t in t_eval:
                    k = int(np.argmin(np.abs(knots * dt - t)))  # nearest knot to t
                    true = xs[starts + knots[k], :3]
                    errs[t].append(np.linalg.norm(X[:, k, :3] - true, axis=-1))
            row = []
            for t in t_eval:
                e = np.concatenate(errs[t]) if errs[t] else np.array([np.nan])
                row.append(f"{100 * np.nanmean(e):7.1f}cm" if np.all(np.isfinite(e)) else "  diverged")
            print(f"{name:<18} {integrator:<8} " + " ".join(row))


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
