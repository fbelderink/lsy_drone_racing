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
}
