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

### Closed-loop results of the long horizon and why it fails

300 episodes each:

| variant | success | t mean | failures |
|---|---|---|---|
| baseline (rerun with refactored code) | 281 (93.7 %) | 6.164 | gate 13, obstacle 3, missed 3 |
| `long_exact_cross` (10×0.02 + 15×0.06 s, exact, crossing gates) | **32 (10.7 %)** | 7.409 | **missed 260** |
| `long_dist05` (same + disturbance estimate) | 49 (16.3 %) | 7.237 | missed 241 |

Screening on 40 seeds (rng 0) to isolate the cause:

| variant | success | t mean | note |
|---|---|---|---|
| `tail` (baseline's 30 fine steps + 10 × 0.06 s) | 3/40 | 6.71 | adding look-ahead *on top* of the baseline breaks it |
| `tail_now` (same without dt weighting) | 14/40 | 6.57 | |
| `uniform50` (50 × 0.02 s, exact) | 24/40 | 6.45 | no coarse steps at all, still fails |
| `long_now` (roadmap grid without dt weighting) | 5/40 | 7.34 | |

So it is neither the coarse steps nor the integrator: **any longer horizon fails with the current
cost.** Two mechanisms, found with `diagnose.py`:

1. **Sampling variance.** The noise accumulates along a rollout, so the spread of the positions (and
   of the tracking cost) across samples grows with time. Undiscounted, the far future dominates the
   cost differences, and the ESS-adapted softmin selects samples by their noisy far future; the
   near-term inputs that decide the next gate are then averaged almost at random. New parameter
   `discount_tau`: running costs are weighted with `exp(−t/τ)`.
2. **Model bias.** On light drones (z bias +1 m/s²) the long-horizon controller stays 0.15–0.2 m
   above the reference and flies over low gates (seed 3: hovered above gate 2, then climbed away).
   The short horizon compensates the bias by fast re-planning; the long one plans against a model
   that is 0.4 m off after 1 s. With the disturbance estimate the z bias converges to +1.1 m/s²
   (oracle 1.13) within about a second and the same episode succeeds in 6.14 s.

Discount screening (40 seeds):

| variant | success | t mean |
|---|---|---|
| `disc03` (baseline + τ = 0.3 s) | 36/40 | 6.22 |
| `disc05` | 37/40 | 6.22 |
| `uniform50_disc03` | 34/40 (from 24) | 6.32 |
| `uniform50_disc05` | 31/40 | 6.36 |
| `long_disc03` | 28/40 (from 5) | 6.77 |
| `long_disc05` | 18/40 | 6.98 |
| `long_dist05_disc03` | 32/40 | 6.68 |

**Dependency confirmed**: the long horizon only works with the discount *and* the disturbance
estimate. Each alone recovers part of the loss. The full matrix is benchmarked with 300 episodes
below.

### 300-episode matrix (steps 1–5 single features and their combinations)

Paired against the baseline (281/300, 6.164 s). Δt is the paired mean over the episodes both
succeed in. Noise floor: about ±2 successes per 100 seeds and ±0.005 s paired.

| variant | success | t mean | Δt paired | ms/step | failures (gate/miss/obst/other) |
|---|---|---|---|---|---|
| baseline | 281 (93.7 %) | 6.164 | — | 13.9 | 13/3/3/0 |
| `nominal` | 283 (94.3 %) | 6.162 | −0.004 | 11.8 | 11/4/2/0 |
| `exact` (ZOH integrator, same grid) | **287 (95.7 %)** | 6.184 | +0.014 | 14.5 | 6/4/3/0 |
| `crossing` (crossing-only gate check) | 226 (75.3 %) | 6.110 | −0.034 | 12.2 | 73/0/1/0 |
| `disc03` | 282 (94.0 %) | 6.180 | +0.015 | 12.6 | 11/3/4/0 |
| `dist02` | 286 (95.3 %) | 6.178 | +0.011 | 12.0 | 8/3/3/0 |
| `dist05` | 281 (93.7 %) | 6.188 | +0.025 | 13.0 | 11/3/5/0 |
| `dist10` | 283 (94.3 %) | 6.194 | +0.028 | 12.0 | 8/3/6/0 |
| `dist05_z` (z axis only) | 285 (95.0 %) | 6.170 | +0.005 | 12.0 | 9/3/3/0 |
| `disc03_dist05z` | **286 (95.3 %)** | 6.180 | +0.012 | 12.0 | 7/4/3/0 |
| `sigma05` (white noise, half σ) | 267 (89.0 %) | 6.124 | −0.043 | 12.0 | 21/5/7/0 |
| `beta06_s` (β = 0.6, σ scaled) | 263 (87.7 %) | 6.113 | −0.052 | 11.4 | 18/13/5/1 |
| `hybrid` (tracking + progress) | 281 (93.7 %) | 6.200 | +0.017 | 19.6 | 11/0/8/0 |
| `contouring` (progress only) | 242 (80.7 %) | 6.560 | +0.397 | 19.1 | 8/0/3/47 timeout |
| `modes4` (4 × 256 samples) | 263 (87.7 %) | 6.296 | +0.140 | 12.4 | 23/10/4/0 |
| `sadapt` (adaptive σ) | 267 (89.0 %) | 6.095 | −0.063 | 12.7 | 25/6/2/0 |
| `modes4_sadapt` | 253 (84.3 %) | 6.260 | +0.094 | 12.5 | 30/8/9/0 |
| `uniform50_disc03` (1.0 s) | 233 (77.7 %) | 6.257 | +0.093 | 22.5 | 14/49/3/1 |
| `u50_disc03_dist05` | 250 (83.3 %) | 6.271 | +0.126 | 22.3 | 8/36/4/2 |
| `u50_disc03_dist05z` | 245 (81.7 %) | 6.235 | +0.078 | 22.4 | 9/40/6/0 |
| `long_dist05_disc03` (1.1 s, coarse tail) | 228 (76.0 %) | 6.602 | +0.460 | 14.4 | 4/62/6/0 |
| `long_dist05z_disc03` | 227 (75.7 %) | 6.584 | +0.422 | 14.4 | 7/62/4/0 |

Reading:

- **Exact discretization helps even on the baseline grid** (+6 successes, fewer gate hits), at a
  cost of +0.014 s. The disturbance estimate helps a little with a slow filter (α = 0.02) or on z
  only; the x/y estimate adds noise (more obstacle hits at α ≥ 0.05). The z-only variant recovers the
  late-reveal seeds 95 and 98 (the z offset of light drones) without losing others.
- **Everything that shrinks the exploration loses gates**: half σ, smoothed noise with scaled σ,
  adaptive σ (which shrinks σ once the ESS is high) and splitting the 1024 samples into 4 modes all
  hit more gate frames but fly faster when they succeed. They make the controller more aggressive
  and less robust. The multi-modal variant with 256 samples per mode has too few samples per mode;
  it depends on a larger sample budget (GPU, step 6).
- **Crossing checks alone hit the frames** (73 gate collisions): a crossing just outside the
  opening and the states near the frame are not penalized enough with a single segment. They are
  only useful together with the slab check ("both"), as used for coarse grids.
- **Contouring alone is too slow and times out**; the hybrid fixes the persistent missed gate of
  seed 37 (all 3 rngs) but loses seed 2 (late-revealed obstacle) and costs 6 ms per step.
- **The long horizons with discount and disturbance estimate still miss gates**: 36–62 missed, most
  at the third pass (gate 3). `diagnose.py --seed=13 --variant=u50_disc03_dist05z`: gate 3 is a
  hairpin, the reference goes through it at 0.5 m/s and immediately loops back up over the frame
  (0.3–0.5 m above the center, inside `gate_frame_limit`). The 1 s horizon sees the whole loop;
  every sample that goes through the opening risks frame cost twice and the time-indexed tracking
  cost gives no reward for actually passing, so the cheapest plan is to stay on the near side,
  climb, and rejoin the reference after the loop. The 0.6 s horizon is already committed when the
  loop comes into view. This is the dependency on the progress cost: the gate bonus of
  `contouring_cost` rewards crossing the next gate (benchmarked next).

### The long horizon needs the progress cost (300 episodes)

| variant | success | t mean | Δt paired | ms/step | failures (gate/miss/obst/other) |
|---|---|---|---|---|---|
| baseline | 281 (93.7 %) | 6.164 | — | 13.9 | 13/3/3/0 |
| `short_best` (exact, nominal, τ = 0.3, z disturbance) | **287 (95.7 %)** | **6.170** | **+0.001** | 13.7 | 5/4/4/0 |
| `short_best_bonus` (+ gate bonus) | 282 (94.0 %) | 6.130 | −0.041 | 21.2 | 7/4/7/0 |
| `short_best_hybrid` (+ full progress cost) | 286 (95.3 %) | 6.146 | −0.024 | 21.2 | 4/3/7/0 |
| `u50_disc03_dist05z` (1.0 s, tracking only) | 245 (81.7 %) | 6.235 | +0.078 | 22.4 | 9/40/6/0 |
| `u50_dd_bonus` (1.0 s + gate bonus) | 287 (95.7 %) | 6.346 | +0.180 | 34.2 | 3/6/4/0 |
| `u50_dd_hybrid` (1.0 s + full progress cost) | **290 (96.7 %)** | 6.467 | +0.297 | 34.2 | 4/3/3/0 |
| `u40_dd` (0.8 s, tracking only) | 276 (92.0 %) | 6.188 | +0.020 | 17.9 | 7/13/4/0 |
| `u40_dd_bonus` (0.8 s + gate bonus) | 286 (95.3 %) | 6.245 | +0.076 | 27.6 | 4/7/3/0 |
| `long_dist05z_disc03` (roadmap grid 1.1 s, tracking only) | 227 (75.7 %) | 6.584 | +0.422 | 14.4 | 7/62/4/0 |
| `long_dd_bonus` (roadmap grid + gate bonus) | 272 (90.7 %) | 6.969 | +0.811 | 20.8 | 5/17/6/0 |
| `long_dd_hybrid` | 278 (92.7 %) | 7.212 | +1.056 | 20.8 | 3/15/4/0 |
| `long_dd_contour` (roadmap grid + pure contouring) | **291 (97.0 %)** | 9.116 | +2.946 | 20.3 | 1/5/2/1 |

`BONUS_ONLY` sets the contour, progress and speed-cap weights to 0, so only the gate bonus is
added to the tracking cost.

Reading:

- **Dependency confirmed: the gate bonus fixes the long horizon** (1.0 s: 245 → 287, 0.8 s: 276 →
  286), while on the 0.6 s horizon it brings nothing (287 → 282). The full progress cost on top of
  the 1.0 s horizon gives the highest success rate at a usable speed (290), at +0.3 s and 2.5× the
  compute per step.
- **The coarse roadmap grid is slower** than a uniform 1.0 s grid (+0.8 s with the bonus): the
  0.06 s steps hold a command three times as long, so the rollouts cannot represent the quick
  attitude changes and the optimum becomes conservative. With pure contouring it is the most robust
  configuration overall (291), but 3 s slower; it is a safe mode, not a racing mode.
- **`short_best` is the best trade-off**: +6 successes against the baseline at the same lap time and
  compute. The gains come from the light-drone z offset (seeds 79, 95, 98) and seed 7.

## Speed analysis — what limits the lap time (`benchmarks/mppi/speed_limits.py`)

Other teams reach laps below 4 s; we fly 6.2 s. Measured with `short_best` on seeds 1–6:

- **The inputs are not saturated.** Commanded |roll| and |pitch| reach at most 0.93 rad (limit
  1.0), the combined tilt 1.0 rad at the 99th percentile, the thrust at most 94 % of its maximum.
  The thrust-to-weight ratio is 1.80, so the drone holds altitude only up to a tilt of 56°, where it
  reaches its maximum level acceleration of 14.7 m/s². **Raising the roll/pitch limit above 1 rad
  (57°) cannot add horizontal acceleration**; it would only trade altitude for it.
- **The reference speed profile sets the lap time.** The drone follows it within a few cm (flown
  time ≈ profile time ≈ 6.2 s). Over the 12.2 m path, 52 % is capped by the 2 m/s limit around not
  yet revealed gates (`v_gate_max`) and 42 % by the lateral acceleration limit of 5 m/s², a third of
  what the drone can do.

Profile lap time on seed 1 for other mid-level limits (offline, nominal gates):

| a_lat | a_long / a_brake | v_gate_max | profile time |
|---|---|---|---|
| 5 | 8 / 7 | 2 | 6.28 s (current) |
| 8 | 8 / 7 | 2 | 5.68 s |
| 5 | 8 / 7 | 3 | 5.54 s |
| 8 | 8 / 7 | 3 | 4.74 s |
| 10 | 10 / 10 | 3 | 4.39 s |
| 10 | 10 / 10 | 4 | 4.08 s |
| 12 | 12 / 12 | 4 | 3.77 s |

Spot check on seed 1: with `v_gate_max` 4 the drone reaches gate 3 too fast for its late reveal and
overshoots it by 0.5 m, also with the 1.0 s horizon; with 3 m/s all gates are passed in ~5.5 s. The
grid below screens the lateral limit against the gate cap in closed loop.

### Speed grid (100 episodes, rng 0, `short_best` with a_long = a_brake = 10)

| a_lat \ v_gate_max | 2.0 m/s | 2.5 m/s | 3.0 m/s |
|---|---|---|---|
| 8 | 95 %, 5.53 s | 78 %, 5.32 s | 66 %, 5.23 s |
| 10 | 69 %, 5.47 s | 60 %, 5.22 s | 49 %, 5.11 s |
| 12 | 46 %, 5.42 s | 42 %, 5.21 s | 24 %, 5.08 s |

`sb_fast3` (a_lat 10, v_gate_max 4, profile 4.08 s) 40 %, 5.00 s; `sb_fast4` (a_lat 12, profile 3.77 s)
16 %, 5.00 s; `u50b_fast2` (1.0 s horizon + gate bonus on a_lat 10, v_gate_max 3) 35 %, 5.48 s.

Two failure modes take over:

- A higher **gate cap** causes gate collisions: the gate's true position is revealed too late to
  correct at 3–4 m/s.
- A higher **lateral limit** causes missed gates: the drone overshoots a turn by 0.3–0.5 m and passes
  just outside `gate_frame_limit` (0.45 m), where the gate cost is zero.

Above a_lat ≈ 9 the profiles promise 4.1–3.8 s but are flown in ~5.0 s. From there the controller is
the limit, not the reference.

Confirmation on 300 episodes (compared with `short_best`, 287, 6.170 s):

| variant | success | t mean | Δt paired | failures (gate/miss/obst) |
|---|---|---|---|---|
| `sb_lat7_vg20` | **285 (95.0 %)** | **5.633** | **−0.538** | 7/4/4 |
| `sb_lat8_vg20` | 281 (93.7 %) | 5.532 | −0.643 | 4/8/7 |
| `sb_lat9_vg20` | 253 (84.3 %) | 5.460 | −0.717 | 4/38/5 |

The overshoot at a_lat 9 (`diagnose.py --seed=4 --variant=sb_lat9_vg20`, last turn into the second
pass of gate 2):

- The gate bonus does not fix it (`sb_lat9_bonus`: seeds 4, 6 and 18 still miss).
- The inputs are not saturated (max |roll| 0.95 rad, thrust 95 %).
- The cause is not isolated yet. Candidates: model error at high tilt and speed, the tracking-cost
  weights at speed, or the attitude response lag. Another lever: the profile caps lateral and
  longitudinal acceleration separately, so a combined limit, a_lat² + a_long² ≤ a_max², would keep
  braking out of the tightest part of a turn.

## Step 6 — GPU

`pixi install -e gpu`, then run the harness or `timing.py` with
`JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false .pixi/envs/gpu/bin/python ...`. The
simulator needs the CPU backend, so `cuda` alone fails. RTX 4080 SUPER.

### Timing on the idle machine (`benchmarks/mppi/timing.py`, 200 steps)

| device | variant | samples | update median | update p95 | full step median |
|---|---|---|---|---|---|
| CPU | baseline | 1024 | 6.5 ms | 7.1 ms | 8.5 ms |
| CPU | baseline | 4096 | 29.1 ms | 35.0 ms | 31.8 ms |
| CPU | default (new) | 1024 | 7.7 ms | 8.3 ms | 9.8 ms |
| CPU | default (new) | 4096 | 36.2 ms | 43.5 ms | 39.2 ms |
| CPU | `u50_dd_hybrid` | 1024 | 19.2 ms | 20.4 ms | 21.8 ms |
| GPU | default (new) | 1024 | 2.9 ms | 3.1 ms | 7.4 ms |
| GPU | default (new) | 4096 | 3.0 ms | 3.2 ms | 7.5 ms |
| GPU | default (new) | 16384 | 3.7 ms | 3.9 ms | 8.2 ms |
| GPU | default (new) | 65536 | 5.7 ms | 6.0 ms | 10.2 ms |
| GPU | default (new) | 262144 | 18.1 ms | 18.5 ms | 22.8 ms |
| GPU | `u50_dd_hybrid` | 1024 / 16384 / 65536 | 5.7 / 7.2 / 11.9 ms | | 10.4 / 11.9 / 16.7 ms |

- The control period is 20 ms. On the CPU only ~1k samples fit, or the 1.0 s hybrid horizon at 1k.
- On the GPU 16k samples cost about the same as 1k, and 65k still fit. The 1.0 s hybrid horizon
  fits with up to 65k samples.
- The full step costs 4.5 ms more than the update on the GPU and 2 ms more on the CPU. That is the
  mid-level planner, the reference sampling and the host/device transfers, which the GPU cannot
  speed up (see item 7).

### Closed loop on the GPU (100 episodes, rng 0, compared with `gpu_sb` = `short_best` on the GPU)

| variant | success | t mean | Δt paired | ms/step |
|---|---|---|---|---|
| `gpu_sb` (1024) | 94 | 6.166 | — | 3.5 |
| `gpu_sb_4k` | **97** | 6.119 | −0.056 | 3.7 |
| `gpu_sb_16k` | **97** | 6.107 | −0.067 | 6.9 |
| `gpu_sb_16k_modes4` (4 × 4096) | 94 | 6.124 | −0.048 | 6.8 |
| `gpu_sb_16k_sadapt` | 91 | 6.049 | −0.107 | 5.2 |
| `gpu_sb_lat8` (1024) | 94 | 5.539 | −0.638 | 3.2 |
| `gpu_sb_16k_lat8` | 90 | 5.472 | −0.703 | 6.4 |
| `gpu_sb_lat10` (1024) | 65 | 5.448 | −0.762 | 3.3 |
| `gpu_sb_16k_lat10` | 56 | 5.388 | −0.857 | 6.5 |
| `gpu_sb_16k_lat8_vg25` | 78 | 5.242 | −0.903 | 6.2 |

Reading:

- **4k samples help a little** (+3 successes, −0.06 s, with no loss on any seed). 16k bring nothing
  more.
- **More samples do not make the fast profiles feasible**: at a_lat 10 the 16k run misses even more
  gates than the 1k run. So the overshoot is not an exploration problem, which points to the
  model, the cost or the profile (see the speed grid above).
- **Multi-modal MPPI recovers with enough samples** (4 × 256: −18 against the baseline on 300
  episodes; 4 × 4096: equal to the 1-mode run) but does not beat one mode with the same samples. On
  this track there is no second homotopy class that the single mode misses. Adaptive spread still
  shrinks σ too much and hits more frames.

## Step 7 — fully jitted closed loop: not done

The mid-level planner (spline planner, speed profile, progress projection) is numpy/scipy code that
runs on the host every step. A fully jitted loop needs it in JAX, which is exactly issue #13 ("Make
High-Level Planner & Mid Level Planner JAX compatible"). This item is left to that issue. The
timing above shows the gain to expect: the host side is 2 ms per step on the CPU and 4.5 ms on the
GPU, more than the 16k-sample update itself. A jitted loop would also allow running thousands of
episodes per minute on the GPU for tuning.

## New defaults

`DEFAULT_PARAMS` and `MLC_CONSTANTS` are now the best configuration (`sb_lat7_vg20`):

- `integrator` "exact", `include_nominal` True, `discount_tau` 0.3, `dist_alpha` 0.05,
  `dist_axes` [0, 0, 1];
- mid-level `a_lat_max` 7, `a_long_max` 10, `a_brake_max` 10.

Every other variant in `variants.py` is now defined relative to the old defaults (`OLD_DEFAULTS`,
`OLD_MLC`). Re-running `check` reproduces the stored baseline exactly, and the new `default` variant
reproduces `sb_lat7_vg20` exactly (285/300, 5.633 s, paired Δ 0.000 s).

**Result: 285/300 (95.0 %) at 5.633 s, against 281/300 (93.7 %) at 6.164 s before: −0.53 s per lap
with 4 more successes, at the same compute (CPU 9.8 ms per step).**

### Note on the benchmark machine

Stopping a benchmark with `pkill` on the parent left its 8 workers running; together with a new
run this exhausted the 31 GB of memory and the OOM killer closed other applications. The harness now
recycles workers every 5 episodes (a long-horizon worker grows to ~3 GB in 10 episodes, mostly
from jit caches of the per-episode controller), workers exit when the benchmark process dies, a
worker waits before an episode while less than 4 GB of memory are available, and at most 6 workers
are used. Benchmarks are stopped by killing their process group.
