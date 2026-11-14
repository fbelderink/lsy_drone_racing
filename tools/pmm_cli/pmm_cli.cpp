// Command-line wrapper around the PMM minimum-time planner (third_party/pmm_uav_planner).
//
// Reads one planning problem from stdin as whitespace-separated numbers:
//   g max_acc_norm max_vel_norm sample_dt
//   start_vel(3) end_vel(3)
//   n_waypoints waypoint_0(3) ... waypoint_{n-1}(3)   (waypoint_0 is the start position)
// and writes the sampled trajectory to stdout, one row per sample:
//   t px py pz vx vy vz
// The planner's own log output is redirected to stderr. Exits with 1 on invalid input or an
// infeasible trajectory.

#include <cstdio>
#include <iostream>
#include <vector>

#include "common.hpp"
#include "pmm_mg_trajectory3d.hpp"

using namespace pmm;

// Velocity optimization settings from the upstream config/planner/planner_config.yaml
static constexpr Scalar DT_PRECISION = 1e-3;
static constexpr int GD_MAX_ITER = 30;
static constexpr Scalar GD_ALPHA = 10.0;
static constexpr Scalar GD_ALPHA_RED_FCTR = 0.2;
static constexpr Scalar GD_ALPHA_MIN_THR = 1e-3;
static constexpr int TD_MAX_ITER = 20;
static constexpr Scalar TD_ACC_PRECISION = 1e-2;
static constexpr int GD_MAX_ITER_2ND = 10;
static constexpr Scalar GD_ALPHA_2ND = 35.0;
static constexpr Scalar GD_ALPHA_RED_FCTR_2ND = 0.1;
static constexpr Scalar GD_ALPHA_MIN_THR_2ND = 1e-2;

static bool read_vector(Vector<3>& v) { return static_cast<bool>(std::cin >> v(0) >> v(1) >> v(2)); }

int main() {
  Scalar g, max_acc_norm, max_vel_norm, sample_dt;
  Vector<3> start_vel, end_vel;
  int n_waypoints;
  if (!(std::cin >> g >> max_acc_norm >> max_vel_norm >> sample_dt) || !read_vector(start_vel) ||
      !read_vector(end_vel) || !(std::cin >> n_waypoints) || n_waypoints < 2) {
    std::cerr << "pmm_cli: invalid header" << std::endl;
    return 1;
  }
  std::vector<Vector<3>> waypoints(n_waypoints);
  for (auto& waypoint : waypoints) {
    if (!read_vector(waypoint)) {
      std::cerr << "pmm_cli: expected " << n_waypoints << " waypoints" << std::endl;
      return 1;
    }
  }

  G = g;
  GVEC = {0, 0, -g};
  D_coeffs.setZero();

  // Keep stdout for the samples only
  std::streambuf* stdout_buffer = std::cout.rdbuf(std::cerr.rdbuf());
  PMM_MG_Trajectory3D trajectory(waypoints, start_vel, end_vel, max_acc_norm, max_vel_norm,
                                 DT_PRECISION, GD_MAX_ITER, GD_ALPHA, GD_ALPHA_RED_FCTR,
                                 GD_ALPHA_MIN_THR, TD_MAX_ITER, TD_ACC_PRECISION, true,
                                 GD_MAX_ITER_2ND, GD_ALPHA_2ND, GD_ALPHA_RED_FCTR_2ND,
                                 GD_ALPHA_MIN_THR_2ND, false, false);
  auto [t, p, v, a] = trajectory.get_sampled_trajectory(sample_dt);
  std::cout.rdbuf(stdout_buffer);

  if (t.empty()) {
    std::cerr << "pmm_cli: no feasible trajectory" << std::endl;
    return 1;
  }
  for (size_t i = 0; i < t.size(); i++) {
    std::printf("%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", t[i], p[i](0), p[i](1), p[i](2), v[i](0),
                v[i](1), v[i](2));
  }
  return 0;
}
