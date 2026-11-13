"""MPPI variants for `scripts/bench_mppi.py`.

Each variant maps to overrides of `DEFAULT_PARAMS` in `lsy_drone_racing/control/attitude_mppi.py`.
The special keys `cost_constants`, `cost_weights` and `mlc` override entries of the controller's
`COST_CONSTANTS`, `COST_WEIGHTS` and `MLC_CONSTANTS` dictionaries.
"""

VARIANTS: dict[str, dict] = {"baseline": {}}
