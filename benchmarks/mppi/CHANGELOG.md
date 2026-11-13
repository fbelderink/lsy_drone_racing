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

---

## Step 1 — cheap features from the reference (smoothed noise, truncated sampling, nominal samples, NaN guard)

### Changes (`attitude_mppi.py`, all off by default)

- `MPPIOptions`: one hashable tuple of static options (sample count and feature switches) passed
  to the jitted update, instead of `n_samples` alone.
- `sample_actions`:
  - `noise_beta`: temporally smoothed noise, an AR(1) process per sample and channel,
    `e_t = beta e_(t-1) + sqrt(1 - beta^2) w_t`. The `sqrt(1 - beta^2)` keeps the marginal standard
    deviation at `sigma`.
  - `truncated`: white noise drawn from a normal truncated to the action limits instead of clipped.
- `include_nominal`: sample 0 is the unperturbed nominal sequence, sample 1 is hovering.
- `nan_guard`: NaN or inf rollout costs are replaced with 1e9.

### Results (300 episodes each)

| variant | success | t mean | Δt paired | gained / lost | failures |
|---|---|---|---|---|---|
| baseline | 281 (93.7 %) | 6.165 | | | gate 12, obstacle 4, missed 3 |
| `nan_guard` | 281 | 6.165 | 0.000 | 0 / 0 | bit-identical: no NaN or inf cost occurs |
| `include_nominal` | **283** | 6.162 | −0.004 ± 0.003 | 7 / 5 | gate 11, obstacle 2, missed 4 |
| β = 0.3 | 278 | 6.165 | +0.001 ± 0.003 | 5 / 8 | missed 6 |
| β = 0.6 | 261 | 6.144 | −0.017 ± 0.004 | 6 / 26 | **missed 23** |
| β = 0.8 | 220 | 6.102 | −0.071 ± 0.006 | 6 / 67 | **missed 68** |
| `truncated` | 271 | **7.200** | **+1.019 ± 0.033** | 7 / 17 | missed 14, gate 13 |

### Interpretation

- **NaN guard**: no effect, because no rollout ever diverges with the current model and step length.
  Kept as insurance for coarse steps and new cost terms (zero cost).
- **Nominal and hover samples**: within the noise floor, slightly positive. Kept: cheap and it
  guarantees that the update can always keep the current plan.
- **Smoothed noise** makes the drone faster on the seeds it finishes (−0.07 s at β = 0.8) but fails
  more and more by flying past gates (missed gate, no contact). With the marginal `sigma` kept, the
  variance of the noise summed over the horizon grows by (1 + β) / (1 − β), i.e. 4× at β = 0.6 and
  9× at β = 0.8. The rollouts fan out much further, fewer samples stay near the gate, and the mean
  is a mix of far-apart paths. **Dependency**: β and `sigma` have to be tuned together. Step 1b
  tests β with `sigma` scaled by sqrt((1 − β) / (1 + β)). Smoothed noise may also pay off only with
  a longer horizon (where a few coarse, smooth inputs have to cover 1 s) or with more samples.
- **Truncated sampling** is 1 s slower (max. speed 2.92 → 2.57 m/s). A normal truncated near a limit
  has its mean pulled toward the interior. MPPI averages the samples, so this biases every update
  toward level attitude and away from the limits, which the fast maneuvers use (roll and pitch are
  at ±1 rad often with `sigma` = 0.9). Clipping keeps the mean at the limit. Not adopted. It would
  need the update to use the noise relative to the truncated mean, or a smaller `sigma`.

Adopted for later steps: `include_nominal` and `nan_guard`.

---

## Step 2 — longer horizon with variable timesteps

### Changes (`attitude_mppi.py`)

- `time_grid`: `horizon` steps of 1/freq followed by `n_coarse` steps of `coarse_factor`/freq. The
  coarse steps are integer multiples of the control period, so the mid-level reference is requested
  at the control rate for the whole horizon and sampled at the knots (`_reference`).
- `rollout` scans over (input, step length, per-step discretization).
- `integrator`:
  - `euler`: forward Euler on the full state (original).
  - `exact`: `discretize` computes the zero-order-hold matrix exponential of every attitude axis
    (`[a, ȧ]' = [[0, 1], [k_a, k_r]] [a, ȧ] + [0, k_c] u`) and the exact thrust lag gain
    `1 − exp(−k_t dt)` per step on the host, once. Translation uses the trapezoidal rule on the
    accelerations at the start and the end of the step (exact for an acceleration that changes
    linearly within the step).
  - `rk4`: classic Runge–Kutta on the full state, as accuracy reference.
- `shift_controls`: shift the plan by one control period on a non-uniform grid (piecewise-constant
  inputs evaluated at the new knot times; on a uniform grid this is "drop the first, repeat the
  last").
- Running costs (deviation, velocity, gate slab, obstacle) are weighted per state by the length of
  its step in control periods, so a coarse step counts as much as the fine steps it replaces.
- `gate_mode`: `slab` (original: every state within 0.1 m of a gate plane must be near the center),
  `crossing` (every segment between two states that crosses a gate plane is checked at the
  intersection point, scaled by `gate_crossing_scale` = 4 ≈ number of states in the slab at
  2.5 m/s), `both`. At 0.06 s steps and 3 m/s a state moves 0.18 m, so the 0.2 m thick slab can be
  jumped over; the crossing check cannot.

### Open-loop model accuracy (`benchmarks/mppi/model_accuracy.py`, seeds 1–3)

Position error of the planning model against the env's `first_principles` sim, predicted from every
5th state of a closed-loop baseline episode with the commands that were applied (held over a coarse
step as the planner does). `exact+bias` adds the per-episode constant acceleration bias fitted
from one-step residuals (an oracle for the disturbance estimate of step 3).

| grid | integrator | 0.3 s | 0.6 s | 1.0 s |
|---|---|---|---|---|
| uniform 0.02 s | euler | 4.6 cm | 17.2 cm | 43.7 cm |
| uniform 0.02 s | exact | 4.4 cm | 16.8 cm | 41.7 cm |
| uniform 0.02 s | rk4 | 4.4 cm | 16.8 cm | 41.7 cm |
| uniform 0.02 s | exact+bias | 3.0 cm | 10.6 cm | 22.9 cm |
| uniform 0.06 s | euler | 6.6 cm | 21.8 cm | 58.6 cm |
| uniform 0.06 s | exact | 4.8 cm | 17.9 cm | 45.0 cm |
| uniform 0.10 s | euler | 9.5 cm | 32.0 cm | 103.7 cm |
| uniform 0.10 s | exact | 5.5 cm | 21.5 cm | 51.0 cm |
| uniform 0.12 s | euler | 8.0 cm | 37.5 cm | 125.6 cm |
| uniform 0.12 s | exact | 3.6 cm | 23.3 cm | 50.3 cm |
| 10×0.02 + 15×0.06 s | euler | 5.5 cm | 19.5 cm | 48.4 cm |
| 10×0.02 + 15×0.06 s | exact | 5.0 cm | 18.1 cm | 41.0 cm |
| 10×0.02 + 15×0.06 s | exact+bias | 3.4 cm | 11.9 cm | 24.4 cm |

Fitted biases (z): −0.09, +0.94 and +1.13 m/s² for seeds 1, 2, 3 (mass randomization).

- The exact discretization matches RK4 everywhere and is much better than Euler from 0.06 s on
  (2× at 0.10–0.12 s). At 0.02 s all integrators are equal.
- **The model error is dominated by a constant bias, not by the discretization.** The error grows
  quadratically (a constant ~1 m/s² gives 0.5 · 1 · 1² = 0.5 m after 1 s). Removing the bias halves
  the error at 0.6 s and 1.0 s.
- **Dependency**: without the bias estimate the prediction 1 s ahead is off by 41 cm, twice the
  gate's half opening. A long horizon then optimizes the far future against a wrong model. In a
  single-episode check (seed 3, light drone) the 1.1 s horizon flew 0.2–0.45 m above the reference
  and missed a gate, because the 15 coarse steps carry 45 of the 55 cost weight and the far future
  dominates the near-term error. The short horizon hides the bias through fast feedback. So the
  long horizon is tested alone and together with the disturbance estimate.

## Step 3 — disturbance estimate

### Changes (`attitude_mppi.py`)

- After each command, the model predicts the next velocity (one fine step with the current
  estimate, `_predict_vel`). At the next observation the estimate is updated with the part that is
  not yet explained, `d += alpha · (v_obs − v_pred) / dt`, clipped to ±`dist_max` per axis. This is
  a first-order low-pass filter of the total residual with time constant dt/alpha (0.4 s at
  alpha = 0.05). `dist_axes` masks the estimated axes.
- The rollouts add `d` to the acceleration (`dist_acc` in `dynamics_step`).
- The estimate is only updated once the drone is 5 cm above its start, because the ground holds the
  drone while the model predicts free fall, which would saturate the estimate before takeoff. It is
  reset every episode (the mass changes).

## Step 4 — progress-based cost

### Changes (`attitude_mppi.py`)

- `cost_mode`: `tracking` (original time-indexed position and velocity tracking), `contouring`
  (replaces it), `hybrid` (both).
- `contouring_cost`, evaluated on a window of the reference path from 0.5 m behind to 6 m ahead of
  the current progress, sampled every 0.05 m (positions, unit tangents, progress, speed profile):
  - contour: squared distance to the path over `contour_tolerance` (0.1 m), per state, weight 0.2.
  - progress: reward of 10 per meter of progress of the last state.
  - speed cap: squared speed above the mid-level speed profile, weight 1. The profile slows down
    before unrevealed gates and in tight turns. As a cap instead of a target, the drone may fly
    slower, but not faster.
  - gate bonus: reward of 20 for crossing the next gate's plane forward within 0.12 m of its center.
- **Shortcut bug found during development**: with an all-points nearest search over the 6 m window,
  a rollout that left the path was credited with the progress of a later section of the track that
  passes close by (the track crosses itself). Pure contouring then veered off before the first gate
  and wandered (3/3 test episodes timed out with 0 gates). Fix: the states are projected one after
  another, each onto the nearest of the 16 path points from 3 behind to 12 ahead of the previous
  state's point, so the progress stays continuous.
