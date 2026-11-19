# MPPI benchmark

`scripts/bench_mppi.py` runs the MPPI controller (`lsy_drone_racing/control/attitude_mppi.py`)
over many seeds in parallel processes. It records the outcome of every episode (success, gate,
obstacle, ground, out of bounds, timeout, missed gate), the gates passed, the lap time and the time
per MPPI update.

```bash
# 300 episodes: seeds 1-100 of level2 x 3 seeds of the sampling noise, ~7 min with 6 workers
python scripts/bench_mppi.py --names=default --n_seeds=100 --rngs=0,1,2
# compare variants seed by seed with stored results
python scripts/bench_mppi.py --names=no_discount,no_dist --n_seeds=100 --rngs=0,1,2 --compare=default
python scripts/bench_mppi.py --show --names=default,no_discount --compare=default
```

Variants are overrides of the controller's parameters, defined in [variants.py](variants.py).
Results are written to `benchmarks/mppi/results/<config>/<variant>.json`.

**Noise floor:** about ±2 successes per 100 episodes. Compare variants on the same seeds; the
paired comparison (`--compare`) reports the time difference on episodes that both variants finish,
and which episodes were gained or lost.

**Machine:** use at most 6 workers. Each worker holds a jitted controller, and the harness waits
before an episode while less than 4 GB of memory are free. Stop a run by killing its process group,
not only the parent.

## Defaults and their effect (300 episodes, level2)

| | success | mean lap time |
|---|---|---|
| before | 281/300 (93.7 %) | 6.16 s |
| now | 285/300 (95.0 %) | 5.63 s |

What the defaults consist of:
- **Faster speed profile:** lateral acceleration 7 m/s² instead of 5, acceleration and braking
  10 m/s² instead of 8 and 7. This gives most of the lap-time gain. Above a lateral limit of ~9
  the drone misses gates: the profile then asks for faster changes of the thrust direction
  (~1,500 °/s) than the drone can make (~500 °/s).
- **Exact discretization of the model:** attitude and thrust are solved exactly over each step;
  the old Euler steps were inaccurate for the springy attitude response. +6 successes.
- **Disturbance estimate on z:** corrects the height offset of drones lighter or heavier than the
  model. +4 successes. On x and y it only adds noise.
- **Discounted running costs** (`discount_tau` 0.3 s): costs further ahead in the horizon weigh
  less, so the spread of far rollouts does not drown the near-term costs.
- **Nominal and hover samples** are always among the samples (neutral to +2).
