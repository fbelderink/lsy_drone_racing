"""Time one control step of MPPI variants on an otherwise idle machine.

Run as:

    $ python benchmarks/mppi/timing.py --names=baseline,hybrid --n_samples=1024,4096

Builds the controller for every variant (and sample count) from the first observation of seed 1 and
calls `compute_control` repeatedly. Reports the median and 95th percentile of the time of the jitted
MPPI update (`update`) and of the whole control step including the mid-level planner and the host
side preparation (`step`). Set `JAX_PLATFORMS=cuda` to time on the GPU (requires the `gpu` pixi
environment).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(Path(__file__).parent))


def main(
    names: str | tuple = "baseline",
    n_samples: str | int | tuple = "",
    n_steps: int = 100,
    config: str = "level2.toml",
):
    """Print update and step times for the given variants."""
    import gymnasium
    import jax
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
    from variants import VARIANTS

    from lsy_drone_racing.utils import load_config, load_controller

    names = names.split(",") if isinstance(names, str) else list(names)
    sample_counts = [int(n) for n in str(n_samples).strip("()").split(",") if str(n).strip()]
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
            seed=1,
        )
    )
    obs, info = env.reset()
    cls = load_controller(ROOT / "lsy_drone_racing/control/attitude_mppi.py")
    print(f"device: {jax.devices()[0]}")
    print(f"{'variant':<28} {'samples':>8} {'update med':>11} {'p95':>7} {'step med':>9}")
    for name in names:
        for n in sample_counts or [None]:
            params = dict(VARIANTS[name])
            if n is not None:
                params["n_samples"] = n
            cfg.controller.mppi = params
            ctrl = cls(obs, info, cfg)
            steps = []
            for _ in range(n_steps):
                t0 = time.perf_counter()
                ctrl.compute_control(obs, info)
                steps.append(1e3 * (time.perf_counter() - t0))
            upd = np.asarray(ctrl._step_ms[5:])
            print(
                f"{name:<28} {ctrl._n_samples:>8} {np.median(upd):9.2f}ms "
                f"{np.percentile(upd, 95):5.2f}ms {np.median(steps[5:]):7.2f}ms"
            )


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
