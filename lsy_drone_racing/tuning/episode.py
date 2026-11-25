"""One seeded simulation episode with controller parameter overrides, as in `scripts/sim.py`."""

from __future__ import annotations

from pathlib import Path

import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.tuning.reward import EpisodeResult
from lsy_drone_racing.utils import load_config, load_controller

REPO = Path(__file__).parents[2]


def run_episode(
    config_name: str, controller_name: str, seed: int, overrides: dict
) -> EpisodeResult:
    """Run one episode without rendering.

    The seed and the overrides are set in the loaded config in memory; the config file is not
    changed. The episode's result depends only on the seed and the overrides.

    Args:
        config_name: Config file name in `config/`.
        controller_name: Controller file name in `lsy_drone_racing/control/`.
        seed: Seed of the environment's randomization.
        overrides: Controller parameters by name, passed as `config.controller.params`.

    Returns:
        The outcome of the episode.
    """
    config = load_config(REPO / "config" / config_name)
    config.sim.render = False
    config.env.seed = seed
    config.controller.params = overrides
    controller_cls = load_controller(REPO / "lsy_drone_racing/control" / controller_name)
    env = JaxToNumpy(
        gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
        )
    )
    obs, info = env.reset()
    controller = controller_cls(obs, info, config)
    step = 0
    while True:
        action = controller.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = controller.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated or done:
            break
        step += 1
    env.close()
    n_gates = len(obs["gate_sequence"])
    return EpisodeResult(
        seed=seed,
        finished=bool(obs["n_gates_passed"] == n_gates),
        gates_passed=int(obs["n_gates_passed"]),
        n_gates=n_gates,
        lap_time=step / config.env.freq,
    )
