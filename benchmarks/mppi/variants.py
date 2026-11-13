"""MPPI variants for `scripts/bench_mppi.py`.

Each variant maps to overrides of `DEFAULT_PARAMS` in `lsy_drone_racing/control/attitude_mppi.py`.
The special keys `cost_constants`, `cost_weights` and `mlc` override entries of the controller's
`COST_CONSTANTS`, `COST_WEIGHTS` and `MLC_CONSTANTS` dictionaries.
"""

# 10 x 0.02 s + 15 x 0.06 s = 1.1 s with exact discretization and crossing-based gate checks
LONG = {
    "horizon": 10,
    "n_coarse": 15,
    "coarse_factor": 3,
    "integrator": "exact",
    "gate_mode": "both",
}

# Discount (rollout variance) and z disturbance estimate (model bias), needed by long horizons
DD = {"discount_tau": 0.3, "dist_alpha": 0.05, "dist_axes": [0.0, 0.0, 1.0]}
U50_DD = {"horizon": 50, "integrator": "exact", **DD}
LONG_DD = {**LONG, **DD}
SHORT_BEST = {"integrator": "exact", "include_nominal": True, **DD}
# Hybrid cost with only the gate bonus of the progress cost added to tracking
BONUS_ONLY = {"contour_weight": 0.0, "progress_weight": 0.0, "speed_cap_weight": 0.0}

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "check": {},  # identical to baseline, used to verify that a refactor keeps the results
    # Step 1: cheap features from the reference implementation, each alone
    "beta03": {"noise_beta": 0.3},
    "beta06": {"noise_beta": 0.6},
    "beta08": {"noise_beta": 0.8},
    "truncated": {"truncated": True},
    "nominal": {"include_nominal": True},
    "nan_guard": {"nan_guard": True},
    # Step 1b: smoothed noise with sigma scaled by sqrt((1 - beta) / (1 + beta)), which keeps the
    # spread of the summed noise (and so of the positions) about the same as for white noise
    "sigma05": {"sigma": [0.45, 0.45, 0.0, 0.15]},  # white noise with half sigma, as a control
    "beta03_s": {"noise_beta": 0.3, "sigma": [0.66, 0.66, 0.0, 0.22]},
    "beta06_s": {"noise_beta": 0.6, "sigma": [0.45, 0.45, 0.0, 0.15]},
    # Step 2: integrators and longer horizons with coarse steps
    "exact": {"integrator": "exact"},
    "crossing": {"gate_mode": "crossing"},
    # 10 x 0.02 s + 15 x 0.06 s = 1.1 s, the grid proposed in the roadmap
    "long_euler": {"horizon": 10, "n_coarse": 15, "coarse_factor": 3},
    "long_exact": {"horizon": 10, "n_coarse": 15, "coarse_factor": 3, "integrator": "exact"},
    "long_exact_cross": {
        "horizon": 10,
        "n_coarse": 15,
        "coarse_factor": 3,
        "integrator": "exact",
        "gate_mode": "both",
    },
    # 0.8 s: 10 x 0.02 s + 10 x 0.06 s
    "mid_exact_cross": {
        "horizon": 10,
        "n_coarse": 10,
        "coarse_factor": 3,
        "integrator": "exact",
        "gate_mode": "both",
    },
    # Step 3: disturbance estimate
    "dist02": {"dist_alpha": 0.02},
    "dist05": {"dist_alpha": 0.05},
    "dist10": {"dist_alpha": 0.1},
    "dist05_z": {"dist_alpha": 0.05, "dist_axes": [0.0, 0.0, 1.0]},
    # Step 4: progress-based cost
    "contouring": {"cost_mode": "contouring"},
    "hybrid": {"cost_mode": "hybrid"},
    # Combinations: a long horizon needs the disturbance estimate (the model is off by ~0.4 m after
    # 1 s without it), and a progress cost may need a long horizon to see the next turn
    "long_dist05": {**LONG, "dist_alpha": 0.05},
    "long_dist05_hybrid": {**LONG, "dist_alpha": 0.05, "cost_mode": "hybrid"},
    "long_dist05_contour": {**LONG, "dist_alpha": 0.05, "cost_mode": "contouring"},
    # Step 2b: isolate why the long horizon fails
    "tail": {"n_coarse": 10, "coarse_factor": 3, "integrator": "exact", "gate_mode": "both"},
    "tail_now": {
        "n_coarse": 10,
        "coarse_factor": 3,
        "integrator": "exact",
        "gate_mode": "both",
        "step_weighting": "none",
    },
    "uniform50": {"horizon": 50, "integrator": "exact"},
    "long_now": {**LONG, "step_weighting": "none"},
    # Discounted running costs
    "disc03": {"discount_tau": 0.3},
    "disc05": {"discount_tau": 0.5},
    "uniform50_disc03": {"horizon": 50, "integrator": "exact", "discount_tau": 0.3},
    "uniform50_disc05": {"horizon": 50, "integrator": "exact", "discount_tau": 0.5},
    "long_disc03": {**LONG, "discount_tau": 0.3},
    "long_disc05": {**LONG, "discount_tau": 0.5},
    "long_dist05_disc03": {**LONG, "dist_alpha": 0.05, "discount_tau": 0.3},
    # The long horizon needs the discount (variance) and the disturbance estimate (bias) together
    "u50_disc03_dist05": {
        "horizon": 50,
        "integrator": "exact",
        "discount_tau": 0.3,
        "dist_alpha": 0.05,
    },
    "u50_disc03_dist05z": {
        "horizon": 50,
        "integrator": "exact",
        "discount_tau": 0.3,
        "dist_alpha": 0.05,
        "dist_axes": [0.0, 0.0, 1.0],
    },
    "long_dist05z_disc03": {
        **LONG,
        "dist_alpha": 0.05,
        "dist_axes": [0.0, 0.0, 1.0],
        "discount_tau": 0.3,
    },
    "disc03_dist05z": {"discount_tau": 0.3, "dist_alpha": 0.05, "dist_axes": [0.0, 0.0, 1.0]},
    # Step 4b: long horizons skip the hairpin at gate 3 (the rollouts see the loop back over the
    # frame and stay on the near side). The progress cost and its gate bonus reward passing.
    "u50_dd_hybrid": {**U50_DD, "cost_mode": "hybrid"},
    "u50_dd_bonus": {**U50_DD, "cost_mode": "hybrid", "cost_weights": BONUS_ONLY},
    "long_dd_hybrid": {**LONG_DD, "cost_mode": "hybrid"},
    "long_dd_bonus": {**LONG_DD, "cost_mode": "hybrid", "cost_weights": BONUS_ONLY},
    "long_dd_contour": {**LONG_DD, "cost_mode": "contouring"},
    # 0.8 s uniform horizon as a middle ground
    "u40_dd": {**U50_DD, "horizon": 40},
    "u40_dd_bonus": {**U50_DD, "horizon": 40, "cost_mode": "hybrid", "cost_weights": BONUS_ONLY},
    # Best short-horizon combination of the individually neutral-to-positive features
    "short_best": SHORT_BEST,
    "short_best_bonus": {**SHORT_BEST, "cost_mode": "hybrid", "cost_weights": BONUS_ONLY},
    "short_best_hybrid": {**SHORT_BEST, "cost_mode": "hybrid"},
    # Step 5: multi-modal MPPI and adaptive spread (same total of 1024 samples)
    "modes2": {"n_modes": 2},
    "modes4": {"n_modes": 4},
    "sadapt": {"sigma_adapt": 0.1},
    "modes4_sadapt": {"n_modes": 4, "sigma_adapt": 0.1},
}
