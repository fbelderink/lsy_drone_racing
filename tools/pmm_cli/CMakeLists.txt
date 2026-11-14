# Builds pmm_cli against the sources of the third_party/pmm_uav_planner submodule. The upstream
# CMakeLists is not used because it hard-codes the system Eigen path.
cmake_minimum_required(VERSION 3.10)
project(pmm_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(PMM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pmm_uav_planner)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

add_executable(pmm_cli
  pmm_cli.cpp
  ${PMM_DIR}/src/common.cpp
  ${PMM_DIR}/src/pmm_trajectory.cpp
  ${PMM_DIR}/src/pmm_trajectory3d.cpp
  ${PMM_DIR}/src/pmm_mg_trajectory3d.cpp
)
target_include_directories(pmm_cli PRIVATE ${PMM_DIR}/include)
target_link_libraries(pmm_cli PRIVATE Eigen3::Eigen yaml-cpp)
