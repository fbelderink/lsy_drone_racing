"""Variants of the MPPI controller for `scripts/bench_mppi.py`.

Every variant is a dict of overrides of `DEFAULT_PARAMS` in `attitude_mppi.py`, plus the optional
sub-dicts `cost_constants` and `cost_weights` that override `COST_CONSTANTS` and
`COST_WEIGHTS`. An empty dict is the controller's defaults.
"""

VARIANTS = {
    "default": {},
    # The features of the defaults, one at a time switched off
    "no_discount": {"discount_tau": 0.0},
    "no_dist": {"dist_alpha": 0.0},
    # More samples, for a GPU (`JAX_PLATFORMS=cuda,cpu`); 1024 fit into 20 ms on a CPU
    "samples_4k": {"n_samples": 4096},
}
