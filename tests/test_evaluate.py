import numpy as np
import pytest

from neural_routing.core import Solution, VRPInstance, evaluate


def line_instance() -> VRPInstance:
    """Depot at 0, customers at positions 1, 2, 3 on a line; 1 time unit per distance unit."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    return VRPInstance(
        name="line",
        coords=np.column_stack([x, np.zeros(4)]),
        durations=np.abs(x[:, None] - x[None]),
        demands=np.array([0.0, 1.0, 1.0, 1.0]),
        service_times=np.array([0.0, 1.0, 1.0, 1.0]),
        time_windows=np.array([[0, np.inf], [5, np.inf], [0, np.inf], [0, 6]]),
        capacity=2.0,
    )


def test_single_route_times_and_violations() -> None:
    m = evaluate(line_instance(), Solution.from_lists([[1, 2, 3]]))
    # arrive 1 -> wait to 5, serve to 6; arrive 2 at 7, serve to 8; arrive 3 at 9 > 6 (3 s late)
    assert m.travel_time_s == 6.0
    assert m.route_duration_s == 10.0 + 3.0
    assert (m.tw_violations, m.lateness_s) == (1, 3.0)
    assert m.capacity_excess == 1.0
    assert m.is_complete and not m.is_feasible


def test_missing_and_duplicates() -> None:
    m = evaluate(line_instance(), Solution.from_lists([[1, 1], [3]]))
    assert (m.missing, m.duplicated) == (1, 1)


def test_invalid_node_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate(line_instance(), Solution.from_lists([[0, 1]]))


def test_shape_validation() -> None:
    inst = line_instance()
    with pytest.raises(ValueError):
        VRPInstance(
            name="bad",
            coords=inst.coords,
            durations=inst.durations[:3],
            demands=inst.demands,
            service_times=inst.service_times,
            time_windows=inst.time_windows,
            capacity=1.0,
        )


def test_save_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    inst = line_instance()
    inst.save(tmp_path / "x.npz")
    back = VRPInstance.load(tmp_path / "x.npz")
    np.testing.assert_array_equal(back.time_windows, inst.time_windows)
    assert back.capacity == inst.capacity
