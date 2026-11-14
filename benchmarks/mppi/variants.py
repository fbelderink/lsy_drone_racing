"""MPPI variants for `scripts/bench_mppi.py`.

Each variant maps to overrides of `DEFAULT_PARAMS` in `lsy_drone_racing/control/attitude_mppi.py`.
All variants except `default` are defined relative to the original defaults (`OLD_DEFAULTS`), so
their results stay reproducible after the defaults were changed to the best configuration.
The special keys `cost_constants`, `cost_weights` and `mlc` override entries of the controller's
`COST_CONSTANTS`, `COST_WEIGHTS` and `MLC_CONSTANTS` dictionaries.
"""

# The defaults before the roadmap work, applied below to every variant except `default`
OLD_DEFAULTS = {
    "integrator": "euler",
    "include_nominal": False,
    "discount_tau": 0.0,
    "dist_alpha": 0.0,
    "dist_axes": [1.0, 1.0, 1.0],
}
OLD_MLC = {"a_long_max": 8.0, "a_lat_max": 5.0, "a_brake_max": 7.0}

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

# Faster reference speed profiles (mid-level controller limits); default: a_lat 5, a_long 8,
# a_brake 7, v_gate_max 2
FAST = {
    1: {"a_lat_max": 8.0, "v_gate_max": 3.0},
    2: {"a_lat_max": 10.0, "a_long_max": 10.0, "a_brake_max": 10.0, "v_gate_max": 3.0},
    3: {"a_lat_max": 10.0, "a_long_max": 10.0, "a_brake_max": 10.0, "v_gate_max": 4.0},
    4: {"a_lat_max": 12.0, "a_long_max": 12.0, "a_brake_max": 12.0, "v_gate_max": 4.0},
}

LAT8 = {"a_lat_max": 8.0, "a_long_max": 10.0, "a_brake_max": 10.0, "v_gate_max": 2.0}
LAT10 = {**LAT8, "a_lat_max": 10.0}

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
    # Speed: the lap time is set by the reference speed profile, not by the input limits (see
    # `speed_limits.py`). Profile lap times on seed 1: 6.28 s now, 4.74 / 4.39 / 4.08 / 3.77 s.
    "sb_fast1": {**SHORT_BEST, "mlc": FAST[1]},
    "sb_fast2": {**SHORT_BEST, "mlc": FAST[2]},
    "sb_fast3": {**SHORT_BEST, "mlc": FAST[3]},
    "sb_fast4": {**SHORT_BEST, "mlc": FAST[4]},
    "u50b_fast2": {**U50_DD, "cost_mode": "hybrid", "cost_weights": BONUS_ONLY, "mlc": FAST[2]},
    "u50b_fast3": {**U50_DD, "cost_mode": "hybrid", "cost_weights": BONUS_ONLY, "mlc": FAST[3]},
    "u50b_fast4": {**U50_DD, "cost_mode": "hybrid", "cost_weights": BONUS_ONLY, "mlc": FAST[4]},
    # Speed grid: lateral acceleration x gate cap, with a_long = a_brake = 10
    **{
        f"sb_lat{a}_vg{str(vg).replace('.', '')}": {
            **SHORT_BEST,
            "mlc": {
                "a_lat_max": float(a),
                "a_long_max": 10.0,
                "a_brake_max": 10.0,
                "v_gate_max": vg,
            },
        }
        for a in (7, 8, 9, 10, 12)
        for vg in (2.0, 2.5, 3.0)
    },
    # Step 5: multi-modal MPPI and adaptive spread (same total of 1024 samples)
    "modes2": {"n_modes": 2},
    "modes4": {"n_modes": 4},
    "sadapt": {"sigma_adapt": 0.1},
    "modes4_sadapt": {"n_modes": 4, "sigma_adapt": 0.1},
    # Step 6: GPU closed loop (run with the gpu environment and JAX_PLATFORMS=cuda; the results
    # differ numerically from the CPU ones, so they are only compared with each other)
    "gpu_sb": SHORT_BEST,
    "gpu_sb_4k": {**SHORT_BEST, "n_samples": 4096},
    "gpu_sb_16k": {**SHORT_BEST, "n_samples": 16384},
    "gpu_sb_16k_modes4": {**SHORT_BEST, "n_samples": 16384, "n_modes": 4},
    "gpu_sb_16k_sadapt": {**SHORT_BEST, "n_samples": 16384, "sigma_adapt": 0.1},
    # GPU: does a larger sample budget allow the faster profiles?
    "gpu_sb_lat8": {**SHORT_BEST, "mlc": LAT8},
    "gpu_sb_16k_lat8": {**SHORT_BEST, "n_samples": 16384, "mlc": LAT8},
    "gpu_sb_lat10": {**SHORT_BEST, "mlc": LAT10},
    "gpu_sb_16k_lat10": {**SHORT_BEST, "n_samples": 16384, "mlc": LAT10},
    "gpu_sb_16k_lat8_vg25": {**SHORT_BEST, "n_samples": 16384, "mlc": {**LAT8, "v_gate_max": 2.5}},
    # At speed the drone drifts wide and passes outside `gate_frame_limit`, where the gate cost is
    # zero; the gate bonus rewards passing (it did not help at the default speed)
    "sb_lat8_bonus": {**SHORT_BEST, "mlc": LAT8, "cost_mode": "hybrid", "cost_weights": BONUS_ONLY},
    "sb_lat9_bonus": {
        **SHORT_BEST,
        "mlc": {**LAT8, "a_lat_max": 9.0},
        "cost_mode": "hybrid",
        "cost_weights": BONUS_ONLY,
    },
    "sb_lat10_bonus": {
        **SHORT_BEST,
        "mlc": LAT10,
        "cost_mode": "hybrid",
        "cost_weights": BONUS_ONLY,
    },
}

# The MPPI obstacle tube (0.2 m) is wider than the spline planner's clearance (0.141 m: drone 0.086
# + pole 0.015 + margin 0.04), so the cost pushes the plan off reference paths that pass a pole
# closely; at speed the drone then overshoots the next gate
LAT9 = {**LAT8, "a_lat_max": 9.0}
for _r in (0.12, 0.15):
    _cc = {"obstacle_radius": _r}
    _tag = f"r{int(_r * 100)}"
    VARIANTS[f"sb_{_tag}"] = {
        **SHORT_BEST,
        "mlc": {**LAT8, "a_lat_max": 7.0},
        "cost_constants": _cc,
    }
    VARIANTS[f"sb_lat8_{_tag}"] = {**SHORT_BEST, "mlc": LAT8, "cost_constants": _cc}
    VARIANTS[f"sb_lat9_{_tag}"] = {**SHORT_BEST, "mlc": LAT9, "cost_constants": _cc}
    VARIANTS[f"sb_lat10_{_tag}"] = {**SHORT_BEST, "mlc": LAT10, "cost_constants": _cc}
    VARIANTS[f"sb_lat12_{_tag}"] = {
        **SHORT_BEST,
        "mlc": {**LAT8, "a_lat_max": 12.0},
        "cost_constants": _cc,
    }

# At speed the turn first appears in the discounted far end of the horizon (weight exp(-t / tau))
for _tau in (0.0, 0.6, 1.0):
    _tag = f"tau{int(_tau * 10):02d}"
    VARIANTS[f"sb_lat9_{_tag}"] = {**SHORT_BEST, "mlc": LAT9, "discount_tau": _tau}
    VARIANTS[f"sb_lat10_{_tag}"] = {**SHORT_BEST, "mlc": LAT10, "discount_tau": _tau}

# Thrust-feasible speed profile: |a + g| and its tilt within the drone's limits instead of a lateral
# acceleration limit that ignores gravity (a_lat_max 20 leaves the thrust limits in charge)
_G = 9.81
FEAS = {
    "16": {"thrust_max": 1.6 * _G, "thrust_min": 0.3 * _G, "tilt_max": 0.96},  # 55 deg
    "15": {"thrust_max": 1.5 * _G, "thrust_min": 0.4 * _G, "tilt_max": 0.87},  # 50 deg
    "17": {"thrust_max": 1.7 * _G, "thrust_min": 0.2 * _G, "tilt_max": 1.01},  # 58 deg
}
for _k, _f in FEAS.items():
    VARIANTS[f"sb_feas{_k}"] = {**SHORT_BEST, "mlc": {**LAT8, "a_lat_max": 20.0, **_f}}
    VARIANTS[f"sb_feas{_k}_a12"] = {
        **SHORT_BEST,
        "mlc": {**LAT8, "a_lat_max": 20.0, "a_long_max": 12.0, "a_brake_max": 12.0, **_f},
    }

# Jerk limit (`smooth_time`): a switch between accelerating and braking turns the thrust direction
# by up to ~90 degrees, which the drone needs ~0.2-0.3 s for; the profiles above switch instantly
LAT7 = {**LAT8, "a_lat_max": 7.0}
for _w in (0.1, 0.15, 0.25):
    _tag = f"s{int(round(_w * 100)):02d}"
    VARIANTS[f"sb_lat7_{_tag}"] = {**SHORT_BEST, "mlc": {**LAT7, "smooth_time": _w}}
    VARIANTS[f"sb_lat9_{_tag}"] = {**SHORT_BEST, "mlc": {**LAT9, "smooth_time": _w}}
    VARIANTS[f"sb_lat12_{_tag}"] = {
        **SHORT_BEST,
        "mlc": {
            **LAT8,
            "a_lat_max": 12.0,
            "a_long_max": 12.0,
            "a_brake_max": 12.0,
            "smooth_time": _w,
        },
    }
    VARIANTS[f"sb_feas16_{_tag}"] = {
        **SHORT_BEST,
        "mlc": {**LAT8, "a_lat_max": 20.0, **FEAS["16"], "smooth_time": _w},
    }
    VARIANTS[f"sb_feas17_a12_{_tag}"] = {
        **SHORT_BEST,
        "mlc": {
            **LAT8,
            "a_lat_max": 20.0,
            "a_long_max": 12.0,
            "a_brake_max": 12.0,
            **FEAS["17"],
            "smooth_time": _w,
        },
    }

# Short jerk-limit windows on fast profiles (the 0.1-0.25 s windows cost more than they gain).
for _w in (0.05, 0.075, 0.1):
    _tag = f"s{int(round(_w * 1000)):03d}"
    for _a in (12, 15):
        VARIANTS[f"sb_lat{_a}_{_tag}"] = {
            **SHORT_BEST,
            "mlc": {
                **LAT8,
                "a_lat_max": float(_a),
                "a_long_max": float(_a),
                "a_brake_max": float(_a),
                "smooth_time": _w,
            },
        }

# Gate cap sweep on the default profile, for the gate reveal analysis (`gate_reveal.py`)
VARIANTS["sb_lat7_vg40"] = {**SHORT_BEST, "mlc": {**LAT7, "v_gate_max": 4.0}}
VARIANTS["sb_lat7_nocap"] = {**SHORT_BEST, "mlc": {**LAT7, "v_gate_max": 6.0}}

# Per-gate cap from the worst-case correction at the reveal (reaction 0.15 s measured with
# gate_reveal.py), against uniform caps of the same mean
for _q in (0.9, 0.95, 0.98, 1.0):
    VARIANTS[f"sb_gcap_q{int(round(_q * 100))}"] = {
        **SHORT_BEST,
        "mlc": {**LAT7, "gate_cap": {"reaction": 0.15, "q": _q}},
    }
VARIANTS["sb_lat7_vg225"] = {**SHORT_BEST, "mlc": {**LAT7, "v_gate_max": 2.25}}
VARIANTS["sb_lat7_vg250"] = {**SHORT_BEST, "mlc": {**LAT7, "v_gate_max": 2.5}}

# Reference-free gate progress cost (`cost_mode` "gates"): the racing line follows from progress
# along the chain of the next gates. The horizon has to reach two gates ahead (up to ~4-5 m).
GATES = {**SHORT_BEST, "cost_mode": "gates", "mlc": LAT7}
VARIANTS["gates_h12"] = {**GATES, "n_coarse": 10}  # 0.6 s fine + 10 x 0.06 s = 1.2 s
VARIANTS["gates_h18"] = {**GATES, "n_coarse": 20}  # 1.8 s
VARIANTS["gates_h18_tau1"] = {**GATES, "n_coarse": 20, "discount_tau": 1.0}
VARIANTS["gates_h18_tau0"] = {**GATES, "n_coarse": 20, "discount_tau": 0.0}
VARIANTS["gates_h18_nocap"] = {
    **GATES,
    "n_coarse": 20,
    "cost_weights": {"reveal_speed_weight": 0.0},
}

# MPCC racing line: progress along the spline as a loose corridor, no speed profile and no
# reference timing; the line within the corridor and the speed are free
MPCC = {
    **SHORT_BEST,
    "cost_mode": "contouring",
    "mlc": LAT7,
    "cost_weights": {"contour_weight": 0.05, "speed_cap_weight": 0.0},
}
VARIANTS["mpcc_h12"] = {**MPCC, "n_coarse": 10}
VARIANTS["mpcc_h18"] = {**MPCC, "n_coarse": 20}
VARIANTS["mpcc_h18_tau1"] = {**MPCC, "n_coarse": 20, "discount_tau": 1.0}
VARIANTS["mpcc_h18_cw02"] = {
    **MPCC,
    "n_coarse": 20,
    "cost_weights": {"contour_weight": 0.2, "speed_cap_weight": 0.0},
}
VARIANTS["gates_h18_f4"] = {**GATES, "n_coarse": 20, "cost_constants": {"gate_funnel": 4.0}}

# Progress weight: with 10 per meter the drone plays safe (8-10 s laps)
for _pw in (30, 100):
    VARIANTS[f"gates_h12_pw{_pw}"] = {
        **GATES,
        "n_coarse": 10,
        "cost_weights": {"gate_progress_weight": float(_pw)},
    }
    VARIANTS[f"gates_h09_pw{_pw}"] = {
        **GATES,
        "n_coarse": 5,
        "cost_weights": {"gate_progress_weight": float(_pw)},
    }
    VARIANTS[f"mpcc_h12_cw02_pw{_pw}"] = {
        **MPCC,
        "n_coarse": 10,
        "cost_weights": {
            "contour_weight": 0.2,
            "speed_cap_weight": 0.0,
            "progress_weight": float(_pw),
        },
    }

# Long reference-free horizons with a GPU sample budget (run on the GPU)
for _n, _tag in ((16384, "16k"), (65536, "64k")):
    VARIANTS[f"gpu_gates_h12_{_tag}"] = {**GATES, "n_coarse": 10, "n_samples": _n}
    VARIANTS[f"gpu_gates_h18_{_tag}"] = {**GATES, "n_coarse": 20, "n_samples": _n}
    VARIANTS[f"gpu_gates_h09_{_tag}"] = {**GATES, "n_coarse": 5, "n_samples": _n}

# Gate tolerance from the real geometry (opening half 0.2 m - drone 0.086 m): the 0.07 m tolerance
# of the frame cost makes every plan through a gate 1-2 s ahead cost more than its progress earns
GATE_GEOM = {"gate_tolerance": 0.1, "gate_bonus_radius": 0.15}
for _nc, _h in ((5, "09"), (10, "12"), (20, "18")):
    VARIANTS[f"gates_h{_h}_geom"] = {**GATES, "n_coarse": _nc, "cost_constants": GATE_GEOM}
    VARIANTS[f"gates_h{_h}_geom_pw30"] = {
        **GATES,
        "n_coarse": _nc,
        "cost_constants": GATE_GEOM,
        "cost_weights": {"gate_progress_weight": 30.0},
    }


def _on_old_defaults(v: dict) -> dict:
    return {**OLD_DEFAULTS, **v, "mlc": {**OLD_MLC, **v.get("mlc", {})}}


VARIANTS = {k: _on_old_defaults(v) for k, v in VARIANTS.items()}
# The current defaults (`short_best` with a_lat 7, i.e. `sb_lat7_vg20`)
VARIANTS["default"] = {}
