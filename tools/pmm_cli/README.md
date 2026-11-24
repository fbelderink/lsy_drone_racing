# pmm_cli

Command-line wrapper around the PMM minimum-time trajectory planner, used by
`lsy_drone_racing/control/planner/pmm.py`. The input and output format is described at the top
of `pmm_cli.cpp`.

The planner itself is the git submodule `third_party/pmm_uav_planner`
(<https://github.com/ctu-mrs/pmm_uav_planner>), used unmodified. It is licensed under
**GPL-3.0**, Copyright (C) 2024 K. Teissing, M. Novosad, R. Pěnička, M. Saska (Faculty of
Electrical Engineering, Czech Technical University in Prague). `pmm_cli` links against it and is
therefore also distributed under GPL-3.0. The Python code only runs it as a separate program.

K. Teissing, M. Novosad, R. Penicka and M. Saska, "Real-Time Planning of Minimum-Time
Trajectories for Agile UAV Flight," IEEE Robotics and Automation Letters, vol. 9, no. 11,
pp. 10351-10358, Nov. 2024, doi: 10.1109/LRA.2024.3471388.

## Build

`tools/setup_pmm.sh` builds it on activation of the `sim` pixi environment. Manually:

```bash
cmake -S tools/pmm_cli -B tools/pmm_cli/build -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build tools/pmm_cli/build
```
