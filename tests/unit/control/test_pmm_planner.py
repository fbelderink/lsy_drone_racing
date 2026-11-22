from pathlib import Path

import numpy as np
import pytest

from lsy_drone_racing.control.mpcc.path import N_PATH_POINTS, carry_progress, path_from_reference
from lsy_drone_racing.control.planner.pmm import (
    PMM_CLI_PATH,
    DensePathReference,
    format_cli_input,
    gate_normals,
    gate_waypoints,
    parse_cli_output,
    run_pmm,
    turnaround_waypoint,
)

GATES_POS = np.array([[1.0, 0.0, 1.0], [2.0, 2.0, 1.0]])
GATES_NORMAL = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
SEQUENCE = np.array([0, 1])
FORWARD = np.array([1, 1])


@pytest.mark.unit
def test_gate_normals_is_local_x_axis():
    quat = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    np.testing.assert_allclose(gate_normals(quat), GATES_NORMAL, atol=1e-12)


@pytest.mark.unit
def test_gate_waypoints_three_points_per_gate_along_normal():
    waypoints, last_direction = gate_waypoints(
        np.zeros(3), GATES_POS, GATES_NORMAL, SEQUENCE, FORWARD, 0, 0.5, offset=0.25
    )
    expected = [
        [0.75, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.25, 0.0, 1.0],
        [2.0, 1.75, 1.0],
        [2.0, 2.0, 1.0],
        [2.0, 2.25, 1.0],
    ]
    np.testing.assert_allclose(waypoints, expected)
    np.testing.assert_allclose(last_direction, [0.0, 1.0, 0.0])


@pytest.mark.unit
def test_gate_waypoints_reversed_and_repeated_gates_skip_passed():
    sequence = np.array([0, 1, 0])
    direction = np.array([1, 1, -1])
    waypoints, last_direction = gate_waypoints(
        np.array([2.0, 0.0, 1.0]), GATES_POS, GATES_NORMAL, sequence, direction, 1, 0.5, offset=0.25
    )
    # Gate 1 forward, a turnaround because gate 0 lies behind gate 1, then gate 0 against its
    # normal. The turnaround is beside gate 1 on gate 0's side (-x).
    np.testing.assert_allclose(waypoints[:3, 1], [1.75, 2.0, 2.25])
    np.testing.assert_allclose(waypoints[3], [1.5, 2.5, 1.0])
    np.testing.assert_allclose(waypoints[4:, 0], [1.25, 1.0, 0.75])
    np.testing.assert_allclose(last_direction, [-1.0, 0.0, 0.0])


@pytest.mark.unit
def test_gate_waypoints_drops_before_point_already_behind_drone():
    pos = np.array([0.85, 0.0, 1.0])  # between the before-point and the gate's center
    waypoints, _ = gate_waypoints(
        pos, GATES_POS, GATES_NORMAL, SEQUENCE, FORWARD, 0, 0.5, offset=0.25
    )
    assert len(waypoints) == 5
    np.testing.assert_allclose(waypoints[0], GATES_POS[0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("next_center", "expected"),
    [
        ([-1.0, 0.0, 2.0], [0.5, 0.0, 1.6]),  # Higher: over the frame
        ([-1.0, -2.0, 1.0], [0.5, -0.6, 1.0]),  # To the right: beside the frame
        ([-1.0, 2.0, 1.0], [0.5, 0.6, 1.0]),  # To the left
        ([-1.0, 0.0, 0.0], [0.5, -0.6, 1.0]),  # Lower: beside, never below the frame
    ],
)
def test_turnaround_waypoint_goes_around_the_frame_towards_the_next_gate(
    next_center: list[float], expected: list[float]
):
    center = np.array([0.0, 0.0, 1.0])
    direction = np.array([1.0, 0.0, 0.0])
    waypoint = turnaround_waypoint(center, direction, np.array(next_center), side_offset=0.6)
    np.testing.assert_allclose(waypoint, expected, atol=1e-12)


@pytest.mark.unit
def test_gate_waypoints_drops_center_point_right_before_the_plane():
    pos = np.array([0.95, 0.0, 1.0])  # 5 cm before the first gate's plane
    waypoints, _ = gate_waypoints(
        pos, GATES_POS, GATES_NORMAL, SEQUENCE, FORWARD, 0, 0.5, offset=0.25
    )
    assert len(waypoints) == 4
    np.testing.assert_allclose(waypoints[0], [1.25, 0.0, 1.0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("pos", "keeps_turnaround"),
    [
        ([1.2, 0.0, 1.0], True),  # Right behind the opening
        ([1.8, 0.0, 1.0], False),  # Far enough behind the gate
        ([1.2, 0.0, 1.6], False),  # Already above the frame
        ([0.8, 0.0, 1.6], False),  # Over the frame and back on the gate's front side
    ],
)
def test_gate_waypoints_keeps_turnaround_of_the_gate_just_passed(
    pos: list[float], keeps_turnaround: bool
):
    # Gate 1 lies behind gate 0, which the drone has just passed
    gates_pos = np.array([[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    gates_normal = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    waypoints, _ = gate_waypoints(
        np.array(pos), gates_pos, gates_normal, SEQUENCE, FORWARD, 1, 0.5, offset=0.25
    )
    turnaround = turnaround_waypoint(gates_pos[0], gates_normal[0], gates_pos[1], 0.5)
    assert len(waypoints) == 3 + keeps_turnaround
    if keeps_turnaround:
        np.testing.assert_allclose(waypoints[0], turnaround)
    np.testing.assert_allclose(waypoints[-3:, 0], [-0.75, -1.0, -1.25])


@pytest.mark.unit
def test_format_cli_input_layout():
    text = format_cli_input(
        np.zeros(3), np.array([1.0, 0, 0]), GATES_POS, np.array([0, 2.0, 0]), 9.81, 13.0, 10.0, 0.01
    )
    lines = text.splitlines()
    np.testing.assert_allclose(
        [float(x) for x in lines[0].split()], [9.81, 13, 10, 0.01, 1, 0, 0, 0, 2, 0]
    )
    assert lines[1] == "3"
    np.testing.assert_allclose([float(x) for x in lines[4].split()], GATES_POS[1])


@pytest.mark.unit
def test_parse_cli_output():
    samples = parse_cli_output("0 0 0 0 0 0 0\n0.5 1 2 3 4 5 6\n")
    np.testing.assert_allclose(samples[1], [0.5, 1, 2, 3, 4, 5, 6])


@pytest.mark.unit
@pytest.mark.parametrize("text", ["0 0 0 0 0 0 0\n", "0 1 2\n3 4 5\n"])
def test_parse_cli_output_rejects_malformed(text: str):
    with pytest.raises(ValueError):
        parse_cli_output(text)


@pytest.mark.unit
def test_run_pmm_missing_binary_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not found"):
        run_pmm("", cli_path=tmp_path / "missing")


@pytest.mark.unit
def test_dense_path_reference_straight_line():
    points = np.zeros((50, 3))
    points[:, 0] = np.linspace(0.0, 2.0, 50) ** 2 / 2  # uneven spacing
    points = np.vstack((points[:1], points))  # repeated point
    reference = DensePathReference(points, version=3)
    assert reference._version == 3
    assert reference.length == pytest.approx(2.0)
    np.testing.assert_allclose(reference.path(1.5), [1.5, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(reference.path(1.0, 1), [1.0, 0.0, 0.0], atol=1e-6)
    assert reference.get_progress(np.array([0.7, 0.3, 0.0])) == pytest.approx(0.7)
    # Only the window around prev_progress is searched (its end is accurate to the point spacing)
    progress = reference.get_progress(np.array([1.9, 0, 0]), 0.5, search_range_s=(0.1, 0.2))
    assert progress == pytest.approx(0.7, abs=0.011)


@pytest.mark.unit
def test_dense_path_reference_resamples_to_fixed_size():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    path = path_from_reference(DensePathReference(points))
    assert path.positions.shape == (N_PATH_POINTS, 3)
    assert float(path.length) == pytest.approx(2.0, abs=1e-3)


@pytest.mark.unit
def test_dense_path_reference_rejects_single_point():
    with pytest.raises(ValueError):
        DensePathReference(np.zeros((3, 3)))


@pytest.mark.integration
@pytest.mark.skipif(not PMM_CLI_PATH.is_file(), reason="pmm_cli is not built")
def test_run_pmm_passes_through_waypoints():
    waypoints = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.2], [0.0, 1.0, 1.0]])
    cli_input = format_cli_input(
        np.array([0.0, 0.0, 1.0]), np.zeros(3), waypoints, np.zeros(3), 9.81, 13.8
    )
    samples = run_pmm(cli_input)
    assert samples[-1, 0] > 0
    assert np.all(np.diff(samples[:, 0]) > 0)
    for waypoint in waypoints:
        assert np.min(np.linalg.norm(samples[:, 1:4] - waypoint, axis=1)) < 0.05


@pytest.mark.unit
def test_carry_progress_to_replanned_path():
    old_points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    old_path = path_from_reference(DensePathReference(old_points))
    # The new path starts where the drone is, 1.5 m along the old one
    new_reference = DensePathReference(np.array([[1.5, 0.0, 0.0], [5.0, 0.0, 0.0]]), version=2)
    assert carry_progress(old_path, 2.0, new_reference) == pytest.approx(0.5, abs=1e-3)
