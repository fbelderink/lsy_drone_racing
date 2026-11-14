#!/usr/bin/env bash
# Builds tools/pmm_cli (the PMM planner's command-line wrapper) on first activation of the env.
set -eo pipefail

if [ -z "$PIXI_PROJECT_ROOT" ]; then
    echo "[Setup PMM] Not running inside a Pixi environment; skipping setup_pmm.sh"
    exit 0
fi

PMM_DIR="${PIXI_PROJECT_ROOT}/third_party/pmm_uav_planner"
CLI_DIR="${PIXI_PROJECT_ROOT}/tools/pmm_cli"

if [ ! -f "${PMM_DIR}/src/pmm_mg_trajectory3d.cpp" ]; then
  echo "[Setup PMM] Fetching the pmm_uav_planner submodule..."
  git -C "${PIXI_PROJECT_ROOT}" submodule update --init third_party/pmm_uav_planner
fi

if [ ! -x "${CLI_DIR}/build/pmm_cli" ]; then
  echo "[Setup PMM] Building pmm_cli..."
  cmake -S "${CLI_DIR}" -B "${CLI_DIR}/build" -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"
  cmake --build "${CLI_DIR}/build" -j"$(nproc)"
fi

echo "[Setup PMM] pmm_cli is ready!"
