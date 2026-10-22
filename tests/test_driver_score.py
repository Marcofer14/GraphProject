import numpy as np
import pytest

from neural_routing.core import Solution, VRPInstance
from neural_routing.core.driver_score import (
    driver_score,
    erp_per_edit,
    normalised_matrix,
    sequence_deviation,
)

a = np.array


def test_sequence_deviation_by_hand() -> None:
    assert sequence_deviation(a([1, 2, 3, 4]), a([1, 2, 3, 4])) == 0.0
    # positions in actual: [1, 0, 2, 3] -> jumps |-1|-1, |2|-1, |1|-1 = 0, 1, 0 -> 2*1/(4*3)
    assert sequence_deviation(a([1, 2, 3, 4]), a([2, 1, 3, 4])) == pytest.approx(1 / 6)
    # the official metric ignores direction: the reversed route has no jumps
    assert sequence_deviation(a([1, 2, 3, 4]), a([4, 3, 2, 1])) == 0.0
    # worst interleaving of 4 stops: positions [0, 3, 1, 2]... jumps 2, 1, 0 -> 2*3/12
    assert sequence_deviation(a([1, 2, 3, 4]), a([1, 4, 2, 3])) == pytest.approx(0.5)


def test_erp_per_edit_by_hand() -> None:
    M = np.zeros((3, 3))
    M[1, 2] = M[2, 1] = 3.0
    # swapping two stops: two substitutions of cost 3 beat any insert/delete (1000 each)
    assert erp_per_edit(a([1, 2]), a([2, 1]), M) == pytest.approx(3.0)
    assert erp_per_edit(a([1, 2]), a([1, 2]), M) == 0.0
    # if substitution were dearer than two gaps, ERP would delete + insert instead
    M[1, 2] = M[2, 1] = 5000.0
    assert erp_per_edit(a([1, 2]), a([2, 1]), M) == pytest.approx(1000.0)


def test_normalised_matrix_is_shifted_z_score() -> None:
    D = np.array([[0.0, 10.0], [30.0, 0.0]])
    N = normalised_matrix(D)
    assert N.min() == 0.0
    np.testing.assert_allclose(N, (D - D.mean()) / D.std() - ((D - D.mean()) / D.std()).min())


def line_instance(reference: tuple[int, ...] | None) -> VRPInstance:
    x = np.arange(6.0)
    return VRPInstance(
        name="line", coords=np.column_stack([x, np.zeros(6)]),
        durations=np.abs(x[:, None] - x[None]) * 60, demands=np.zeros(6), service_times=np.zeros(6),
        time_windows=np.tile([0.0, np.inf], (6, 1)), capacity=np.inf,
        reference_routes=(reference,) if reference else None,
    )  # fmt: skip


def test_benchmark_rows_carry_driver_score_and_routes() -> None:
    import json

    from neural_routing.eval.benchmark import run_instance

    rows = run_instance(line_instance((1, 2, 3, 4, 5)), ["reference", "nearest_neighbor"])
    by_solver = {r["solver"]: r for r in rows}
    assert by_solver["reference"]["driver_score"] == 0.0
    assert json.loads(by_solver["reference"]["routes"]) == [[1, 2, 3, 4, 5]]
    assert by_solver["nearest_neighbor"]["driver_score"] is not None


def test_driver_score_on_instances() -> None:
    inst = line_instance((1, 2, 3, 4, 5))
    assert driver_score(inst, Solution.from_lists([[1, 2, 3, 4, 5]])) == 0.0
    near = driver_score(inst, Solution.from_lists([[2, 1, 3, 4, 5]]))  # neighbours swapped
    far = driver_score(inst, Solution.from_lists([[5, 2, 3, 4, 1]]))  # ends swapped
    assert near is not None and far is not None and 0 < near < far
    # undefined: no driver route, several routes, or a different set of stops
    assert driver_score(line_instance(None), Solution.from_lists([[1, 2, 3, 4, 5]])) is None
    assert driver_score(inst, Solution.from_lists([[1, 2], [3, 4, 5]])) is None
    assert driver_score(inst, Solution.from_lists([[1, 2, 3, 4]])) is None
