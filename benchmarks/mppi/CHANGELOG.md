# MPPI roadmap changelog (issue #14)

Branch `feat/mppi-roadmap`, branched from `dev` at 483f86f. Every entry lists what changed, how it
was measured and what came out. Raw episode records are in `results/<config>/<variant>.json`, the
variant definitions in `variants.py`.

## Measurement protocol

- `python scripts/bench_mppi.py --names=<variants> --n_seeds=100 --rngs=0,1,2` on `level2.toml`.
  Seeds 1–100 × 3 seeds of the MPPI sampling noise = 300 episodes per variant (screening runs use
  `--rngs=0`, 100 episodes).
- 8 worker processes on a Ryzen 7 5800X, each pinned to one physical core and its SMT sibling, JAX on
  CPU. `ms` is the mean wall time of one MPPI update under this load (2 threads per process), so it
  is comparable between variants but higher than on an idle machine.
- An episode with all 5 gates passed is a success. Lap time `t` is averaged over successful episodes
  only. Failures are classified from the last state before termination against the true object
  poses: `gate` (frame or stand), `obstacle` (pole), `ground`, `out_of_bounds`, `missed_gate` (the
  reference ended without all gates passed, i.e. a gate was flown past without touching it),
  `timeout`.
- Runs are bit-exactly reproducible, but any change to the controller perturbs every episode
  chaotically. **Noise floor**: the same controller with three different sampling-noise seeds gave
  95, 93 and 93 / 100 successes and mean times 6.174, 6.142 and 6.178 s. Differences below about
  ±2 successes per 100 episodes or ±0.03 s are noise.

---

## Step 0 — evaluation harness, parameter plumbing, failure diagnosis

### Changes

- `scripts/bench_mppi.py`: parallel benchmark over seeds and sampling-noise seeds with failure
  classification, failed-gate index, ms per update, max and mean speed, and paired comparison
  against another variant (time difference on seeds both finished, seeds gained and lost).
- `benchmarks/mppi/variants.py`: named parameter overrides.
- `benchmarks/mppi/diagnose.py`: replay one seed and print the last steps (position, reference,
  tracking error, ESS, distances to the true gates and obstacles).
- `attitude_mppi.py`:
  - Tunables in `DEFAULT_PARAMS`, overridable through `config.controller.mppi` (unknown keys raise).
    `cost_constants`, `cost_weights` and `mlc` override entries of the three constant dicts.
  - Cost constants and weights are passed into the jitted update as a pytree, so variants need no
    module patching and changing a weight does not recompile.
  - `rng_seed` parameter for the sampling noise.
  - Wall time of every update is recorded in `_step_ms` (after `block_until_ready`).
  - **Bug fix** in `step_callback`: `self._finished | n_passed == n_gates` parses as
    `(self._finished | n_passed) == n_gates`. A controller whose reference had ended kept flying
    until the 30 s truncation, so a missed gate was reported as a timeout.

### Baseline (300 episodes)

| variant | success | t mean | t median | ms | failures |
|---|---|---|---|---|---|
| baseline | 281/300 (93.7 %) | 6.165 s | 6.160 s | 12.5 | gate 12, obstacle 4, missed gate 3 |

Seeds that fail for all three noise seeds: 7, 37, 85, 95, 98.

### Diagnosis

- **Seed 7** (fails at the 2nd gate): the drone passes gate 0 at 2.6 m/s and runs into obstacle 1,
  which stands 0.35 m behind gate 0. The reference itself passes only about 0.23 m from the pole.
- **Seeds 85, 95, 98** (gate frame): when the true gate pose is revealed (0.7 m sensor range, about
  0.3 s before the gate at 2 m/s), the reference jumps by 0.1–0.15 m. The tracking error rises to
  0.12–0.2 m and does not close before the gate. There is also a persistent 7–10 cm offset in z in
  several runs (seed 95 flies 7 cm above the reference for the whole climb). The mass is randomized
  by ±5 g on 43.4 g (±12 % of the weight) and the planning model differs from the `first_principles`
  sim, so a constant force bias is expected. The force disturbance of `level2` is white noise drawn
  every sim step and averages out.
- **Seed 37**: gate 3 is missed without contact, then the reference ends.

Implications for the roadmap: the disturbance estimate (item 3) targets the z offset directly. A
longer horizon (item 2) should help react to the late gate reveal and see the obstacle behind gate 0
earlier.
