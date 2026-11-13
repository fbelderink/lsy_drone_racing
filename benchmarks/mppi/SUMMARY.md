# MPPI roadmap (issue #14): summary

Branch `feat/mppi-roadmap`. The full protocol, every table and every diagnosis are in
[CHANGELOG.md](CHANGELOG.md).

**Measurement protocol:**
- 300 episodes per variant (seeds 1–100 × 3 sampling seeds) on `level2.toml`.
- Variants are compared seed by seed against a reference variant.
- The noise floor is about ±2 successes per 100 seeds.

## Result

| | success | mean lap time | CPU time per step |
|---|---|---|---|
| before (old defaults) | 281/300 (93.7 %) | 6.164 s | 8.5 ms |
| **now (new defaults)** | **285/300 (95.0 %)** | **5.633 s** | 9.8 ms |

The new defaults are the best robust configuration: −0.53 s per lap and 4 more successes, at the
same compute.

## Roadmap checklist

| # | Item | Status | Effect |
|---|---|---|---|
| 0 | Evaluation harness | ✅ | `scripts/bench_mppi.py`, `variants.py`, `diagnose.py` |
| 1 | Smoothed noise, truncated sampling, nominal/hover samples, NaN guard | ✅ all built | Only the nominal sample is kept (neutral to +2). Smoothed noise and truncated sampling lose gates; the NaN guard never triggers. |
| 2 | Longer horizon, variable time steps, exact discretization | ✅ | **Exact discretization is kept (+6).** A longer horizon alone drops to 76–82 %. |
| 3 | Disturbance estimate | ✅ | **Kept on z only** (+4: fixes the height offset of light drones). The x/y estimate adds noise. |
| 4 | Progress / contouring cost | ✅ | No gain on the 0.6 s horizon, **needed by every longer horizon** (see dependencies). |
| 5 | Multi-modal MPPI, adaptive spread | ✅ | Not kept. Multi-modal needs ≥ 4k samples per mode and then only ties with one mode. Adaptive spread shrinks the noise too much. |
| 6 | GPU | ✅ | 16k samples take 3.7 ms per update (1k on the CPU: 7.7 ms). In closed loop 4k samples give +3 % successes; 16k give nothing more. |
| 7 | Fully jitted closed loop | ❌ left to issue #13 | Needs the planners in JAX, which is issue #13. Would save 2–4.5 ms per step of host overhead. |

New, outside the roadmap:
- `discount_tau`: costs further ahead in the rollout weigh less.
- Speed analysis (`speed_limits.py`) and a faster speed profile, which gives most of the lap-time gain.

## Dependencies between features (your question)

Features that looked useless alone turned out to need each other. The clearest chain is the long
horizon:

| 1.0 s horizon with … | success |
|---|---|
| nothing else | 60 % (40-seed screen; fails: noise builds up along the rollouts, and model error grows to 0.4 m after 1 s) |
| + cost discount | 78 % |
| + discount + disturbance estimate | 82–83 % (now it skips the hairpin at gate 3: it sees the loop back over the frame, and tracking gives no reward for passing) |
| + discount + disturbance + **gate bonus** | 96 % |
| + discount + disturbance + **full progress cost** | **97 %**, but +0.3 s per lap and 22 ms per step on the CPU |

**So yes: the MPCC/progress cost only helps together with a long horizon**, and the long horizon
also needs the discount and the disturbance estimate. On the short horizon the progress cost brings
nothing.

Other dependencies found:
- Multi-modal MPPI depends on the sample budget: −18 successes with 4 × 256 samples, equal with
  4 × 4096. It needs the GPU.
- Crossing-based gate checks only work together with the slab check (crossing alone: −55).
- Smoothed noise only works if σ is scaled down, and then it explores too little. Every variant
  that reduces exploration hits more gate frames but flies faster when it succeeds.

## Speed: why we are at 5.6 s and others below 4 s

1. **A larger roll/pitch limit would not help.** The commands never reach the ±1 rad limit. The
   drone has a thrust-to-weight ratio of 1.8, so it holds altitude only up to 56° of tilt. Beyond
   that, more tilt costs altitude instead of adding horizontal acceleration.
2. **The reference speed profile sets the lap time; the drone follows it within centimetres.**
   Before, 52 % of the path was capped at 2 m/s around unrevealed gates and 42 % at a lateral
   acceleration of 5 m/s², a third of what the drone can do. Raising the lateral limit to 7 gives
   the −0.54 s.
3. **Above that, the controller becomes the limit.** On paper, profiles with a lateral limit of
   10–12 and a gate cap of 4 m/s give 4.1–3.8 s, but they are flown in ~5.0 s with 16–65 %
   success:
   - A higher gate cap leads to frame hits: the true gate position is revealed too late.
   - A higher lateral limit leads to missed gates: the drone overshoots turns by 0.3–0.5 m.
   - Neither more samples (16k on the GPU), the gate bonus nor the 1.0 s horizon fixes this, and the
     inputs are not saturated.

## Suggestions for getting faster

In order of expected value:

1. **Find the cause of the overshoot at a lateral limit ≥ 9**, with `diagnose.py --seed=4
   --variant=sb_lat9_vg20`. Candidates:
   - model error at high tilt and speed (check with `model_accuracy.py` on fast episodes);
   - tracking-cost weights that are too soft at speed;
   - attitude lag.
2. **Combined acceleration limit in the speed profile**: a_lat² + a_long² ≤ a_max². Today braking
   and turning are each allowed their full limit at the same time.
3. **A better answer to late-revealed gates than a 2 m/s cap**, e.g. a cap that depends on how far
   the gate can still move, or on the remaining distance at reveal time.
4. **A shorter path**: the 12.2 m spline path may itself be longer than needed. At 3 m/s average,
   4 s needs a path under 12 m.
5. **Issue #13** (JAX planners): a jitted closed loop on the GPU would make large-scale tuning of
   the parameters above cheap (thousands of episodes per minute instead of ~3 per second).

## How to reproduce

```bash
python scripts/bench_mppi.py --names=default --n_seeds=100 --rngs=0,1,2   # ~7 min, 6 workers
python scripts/bench_mppi.py --show --names=baseline,default,u50_dd_hybrid
python benchmarks/mppi/speed_limits.py --seeds=1,2,3 --variant=default
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .pixi/envs/gpu/bin/python benchmarks/mppi/timing.py --names=default --n_samples=1024,16384
```

Use at most 6 workers. A long-horizon worker reaches ~3 GB, and the harness waits while less than
4 GB are free. Stop a benchmark by killing its process group, not just the parent.
