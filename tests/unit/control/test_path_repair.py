import numpy as np
import pytest

from lsy_drone_racing.control.planner.path_repair import (
    CROSSING_OFFSET,
    GATE_PUSH_MARGIN,
    MIN_GATE_OFFSET,
    OBSTACLE_PUSH_MARGIN,
    first_collision,
    insert_repair_waypoints,
    insertion_index,
    push_out_of_gate,
    push_out_of_obstacle,
    repair_path_hit,
    repair_waypoint,
    slide_toward_gate,
)
from lsy_drone_racing.control.utils.collision import (
    GATE_INNER_HALF_WIDTH,
    GATE_OUTER_HALF_WIDTH,
    gate_contact,
    obstacle_clearance,
)

DRONE_RADIUS = 0.086
OBSTACLE_RADIUS = 0.015
# Gate at x = 5, 1 m up, facing +x, far from the paths below unless a test uses it
GATE = np.array([5.0, 0.0, 1.0, 1.0, 0.0, 0.0])
FAR_OBSTACLE = np.array([[10.0, 10.0, 1.5]])


def line(start: list[float], end: list[float], n: int = 101) -> np.ndarray:
    """Points evenly spaced from start to end with shape (n, 3)."""
    return np.linspace(start, end, n)


def collision_args(obstacles: np.ndarray) -> tuple:
    return (GATE[None], obstacles, np.zeros(len(obstacles)), DRONE_RADIUS, OBSTACLE_RADIUS)


@pytest.mark.unit
def test_collision_functions_return_numpy_for_numpy_input():
    points = line([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], 5)
    assert isinstance(obstacle_clearance(points, FAR_OBSTACLE, np.zeros(1), 0.1), np.ndarray)
    assert isinstance(gate_contact(points, GATE[None], DRONE_RADIUS), np.ndarray)


@pytest.mark.unit
def test_first_collision_free_path():
    points = line([0.0, 0.0, 1.0], [4.0, 0.0, 1.0])
    assert first_collision(points, *collision_args(FAR_OBSTACLE)) is None


@pytest.mark.unit
def test_first_collision_returns_middle_of_first_colliding_stretch():
    points = line([0.0, 0.0, 1.0], [4.0, 0.0, 1.0])  # 4 cm spacing
    obstacles = np.array([[1.0, 0.0, 1.5], [3.0, 0.0, 1.5]])
    hit = first_collision(points, *collision_args(obstacles))
    # The first pole is hit symmetrically around x = 1
    np.testing.assert_allclose(points[hit], [1.0, 0.0, 1.0], atol=0.021)


@pytest.mark.unit
def test_first_collision_skips_the_start_near_the_drone():
    points = line([0.0, 0.0, 1.0], [0.1, 0.0, 1.0], 11)
    obstacles = np.array([[0.0, 0.05, 1.5]])
    assert first_collision(points, *collision_args(obstacles)) is None


@pytest.mark.unit
def test_first_collision_finds_gate_frame():
    points = line([4.0, 0.3, 1.0], [6.0, 0.3, 1.0])
    hit = first_collision(points, *collision_args(FAR_OBSTACLE))
    np.testing.assert_allclose(points[hit], [5.0, 0.3, 1.0], atol=0.021)


@pytest.mark.unit
def test_push_out_of_obstacle_keeps_direction_and_height():
    moved = push_out_of_obstacle(np.array([1.1, 0.0, 0.7]), np.array([1.0, 0.0]), 0.5)
    np.testing.assert_allclose(moved, [1.5, 0.0, 0.7])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        # On the side bar, nearer the outer edge: beside the frame
        ([0.0, 0.3, 0.0], [0.0, GATE_OUTER_HALF_WIDTH + DRONE_RADIUS + GATE_PUSH_MARGIN, 0.0]),
        # On the top bar: above the frame
        ([0.0, 0.0, 0.3], [0.0, 0.0, GATE_OUTER_HALF_WIDTH + DRONE_RADIUS + GATE_PUSH_MARGIN]),
        # Just inside the opening's edge: towards the center
        ([0.0, 0.15, 0.0], [0.0, GATE_INNER_HALF_WIDTH - DRONE_RADIUS - GATE_PUSH_MARGIN, 0.0]),
        # On the bottom bar: into the opening, never below the frame
        ([0.0, 0.0, -0.3], [0.0, 0.0, -(GATE_INNER_HALF_WIDTH - DRONE_RADIUS - GATE_PUSH_MARGIN)]),
        # At the stand: sideways, keeping the height
        ([0.0, -0.02, -0.6], [0.0, -(0.05 + DRONE_RADIUS + GATE_PUSH_MARGIN), -0.6]),
    ],
)
def test_push_out_of_gate(offset: list[float], expected: list[float]):
    moved = push_out_of_gate(GATE[:3] + offset, GATE, DRONE_RADIUS)
    # The gate's horizontal axis is cross(normal, z) = -y, so the result is checked in world frame
    np.testing.assert_allclose(moved, GATE[:3] + expected, atol=1e-9)
    assert not gate_contact(moved[None], GATE[None], DRONE_RADIUS)[0]


@pytest.mark.unit
def test_insertion_index_between_the_waypoints_around_the_hit():
    points = line([0.0, 0.0, 1.0], [4.0, 0.0, 1.0])
    waypoints = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]])
    assert insertion_index(points, waypoints, hit=10) == 0
    assert insertion_index(points, waypoints, hit=40) == 1
    assert insertion_index(points, waypoints, hit=90) == 3


@pytest.mark.unit
def test_repair_waypoint_pushes_out_of_the_nearest_obstacle_or_leaves_free_points():
    obstacles = np.array([[1.0, 0.0, 1.5], [3.0, 0.0, 1.5]])
    moved = repair_waypoint(np.array([2.9, 0.0, 1.0]), *collision_args(obstacles))
    push = DRONE_RADIUS + OBSTACLE_RADIUS + OBSTACLE_PUSH_MARGIN
    np.testing.assert_allclose(moved, [3.0 - push, 0.0, 1.0])
    free = np.array([2.0, 0.0, 1.0])
    np.testing.assert_array_equal(repair_waypoint(free, *collision_args(obstacles)), free)


@pytest.mark.unit
def test_obstacles_next_to_the_straight_gate_crossing_are_not_avoided():
    # Straight crossing of the gate at x = 5, with a pole 0.12 m beside the line, 0.2 m before it
    points = line([4.6, 0.0, 1.0], [5.4, 0.0, 1.0])
    obstacles = np.array([[4.8, 0.12, 1.5]])
    assert first_collision(points, *collision_args(obstacles)) is None
    # A gate's before-waypoint next to the pole slides along the normal towards the gate's center,
    # here down to MIN_GATE_OFFSET because the pole is too close to clear
    slid = repair_waypoint(points[25], *collision_args(obstacles))
    np.testing.assert_allclose(slid, [5.0 - MIN_GATE_OFFSET, 0.0, 1.0])
    # The same pole beside a path away from the gate is avoided
    far_points = points - [3.0, 0.0, 0.0]
    far_obstacles = obstacles - [3.0, 0.0, 0.0]
    assert first_collision(far_points, *collision_args(far_obstacles)) is not None


@pytest.mark.unit
def test_slide_toward_gate_stops_at_the_first_clear_point():
    # Before-waypoint 0.3 m in front of the gate, pole 0.1 m beside the line at 0.3 m
    point = np.array([4.7, 0.0, 1.0])
    slid = slide_toward_gate(point, GATE, np.array([[4.7, 0.1]]), np.array([0.2]))
    assert slid[0] > 4.7 + np.sqrt(0.2**2 - 0.1**2) - 1e-9  # Clear of the pole
    assert slid[0] < 5.0 - MIN_GATE_OFFSET
    np.testing.assert_allclose(slid[1:], [0.0, 1.0])


@pytest.mark.unit
def test_repair_path_hit_straightens_an_oblique_crossing_through_the_frame():
    # Crossing the gate at x = 5 diagonally in -x, through the side bar's inner edge
    points = line([5.4, 0.55, 1.0], [4.6, -0.25, 1.0])
    hit = first_collision(points, *collision_args(FAR_OBSTACLE))
    new = repair_path_hit(points, hit, *collision_args(FAR_OBSTACLE))
    assert new.shape == (2, 3)
    # Before and after the plane along -x, in path order, both inside the opening
    np.testing.assert_allclose(new[:, 0], [5.0 + CROSSING_OFFSET, 5.0 - CROSSING_OFFSET])
    np.testing.assert_allclose(new[0, 1:], new[1, 1:])
    assert abs(new[0, 1]) < GATE_INNER_HALF_WIDTH - DRONE_RADIUS


@pytest.mark.unit
def test_repair_path_hit_returns_one_waypoint_outside_the_gate_case():
    points = line([0.0, 0.0, 1.0], [4.0, 0.0, 1.0])
    obstacles = np.array([[1.0, 0.0, 1.5]])
    hit = first_collision(points, *collision_args(obstacles))
    assert repair_path_hit(points, hit, *collision_args(obstacles)).shape == (1, 3)


@pytest.mark.unit
def test_insert_repair_waypoints_adds_a_single_waypoint():
    waypoints = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    new = insert_repair_waypoints(waypoints, 1, np.array([[1.0, 0.5, 1.0]]))
    np.testing.assert_allclose(new, [[0.0, 0.0, 1.0], [1.0, 0.5, 1.0], [2.0, 0.0, 1.0]])


@pytest.mark.unit
def test_insert_repair_waypoints_crossing_replaces_waypoints_on_it():
    # Gate center and an earlier crossing of the same gate, then a waypoint far behind the gate
    center = GATE[:3]
    old_crossing = [center - [CROSSING_OFFSET, 0.0, 0.05], center + [CROSSING_OFFSET, 0.0, -0.05]]
    after = center + [1.0, 0.5, 0.0]
    waypoints = np.array([center, *old_crossing, after])
    crossing = np.array(
        [center - [CROSSING_OFFSET, 0.0, 0.06], center + [CROSSING_OFFSET, 0.0, -0.06]]
    )
    new = insert_repair_waypoints(waypoints, 3, crossing)
    # A single crossing is left, so the path does not run back and forth through the gate
    np.testing.assert_allclose(new, [*crossing, after])
