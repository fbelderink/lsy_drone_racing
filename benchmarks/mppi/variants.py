"""Variants of the MPPI controller for `scripts/bench_mppi.py`.

Every variant is a dict of overrides of `DEFAULT_PARAMS` in `attitude_mppi.py`, plus the optional
sub-dicts `cost_constants`, `cost_weights` and `mlc` that override `COST_CONSTANTS`,
`COST_WEIGHTS` and `MLC_CONSTANTS`. An empty dict is the controller's defaults.
"""

VARIANTS = {
    "default": {},
    # The features of the defaults, one at a time switched off
    "no_discount": {"discount_tau": 0.0},
    "no_dist": {"dist_alpha": 0.0},
    "slow_profile": {"mlc": {"a_long_max": 8.0, "a_lat_max": 5.0, "a_brake_max": 7.0}},
    # More samples, for a GPU (`JAX_PLATFORMS=cuda,cpu`); 1024 fit into 20 ms on a CPU
    "samples_4k": {"n_samples": 4096},
}
