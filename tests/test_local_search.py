import numpy as np
import pytest

from neural_routing.core import Solution, VRPInstance, evaluate
from neural_routing.data.synthetic import random_instance
from neural_routing.solvers import get_solver
from neural_routing.solvers.local_search import improve_routes, polish, split_giant_tour


def asymmetric(inst: VRPInstance, seed: int) -> VRPInstance:
    """Same instance with a random +-30 % asymmetric perturbation of the matrix."""
    rng = np.random.default_rng(seed)
    D = inst.durations * rng.uniform(0.7, 1.3, size=inst.durations.shape)
    np.fill_diagonal(D, 0.0)
    return VRPInstance(**{**inst.__dict__, "durations": D})


def random_routes(inst: VRPInstance, seed: int) -> list[list[int]]:
    """Feasible but poor routes: customers in random order, cut when capacity is reached."""
    rng = np.random.default_rng(seed)
    routes, cur, load = [], [], 0.0
    for c in rng.permutation(np.arange(1, inst.num_nodes)).tolist():
        if load + inst.demands[c] > inst.capacity:
            routes.append(cur)
            cur, load = [], 0.0
        cur.append(c)
        load += inst.demands[c]
    return [*routes, cur]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("sym", [True, False])
def test_never_worse_complete_and_capacity_safe(seed: int, sym: bool) -> None:
    inst = random_instance(40, seed=seed, capacity=60, num_vehicles=10)
    if not sym:
        inst = asymmetric(inst, seed)
    start = random_routes(inst, seed)
    before = evaluate(inst, Solution.from_lists(start))
    routes, moves = improve_routes(inst, start)
    after = evaluate(inst, Solution.from_lists(routes))
    assert moves > 0
    assert after.is_complete and after.capacity_excess == 0
    assert after.travel_time_s < before.travel_time_s
    assert after.num_routes <= before.num_routes


def test_two_opt_uncrosses_a_square() -> None:
    xy = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    inst = VRPInstance(
        name="square",
        coords=xy,
        durations=np.linalg.norm(xy[:, None] - xy[None], axis=-1),
        demands=np.zeros(4),
        service_times=np.zeros(4),
        time_windows=np.tile([0.0, np.inf], (4, 1)),
        capacity=np.inf,
    )
    routes, _ = improve_routes(inst, [[2, 1, 3]])  # crossing tour, length 2 + 2*sqrt(2)
    assert evaluate(inst, Solution.from_lists(routes)).travel_time_s == pytest.approx(4.0)


def test_local_optimum_is_stable() -> None:
    inst = asymmetric(random_instance(25, seed=3, capacity=40, num_vehicles=6), 3)
    once, _ = improve_routes(inst, random_routes(inst, 3))
    twice, moves = improve_routes(inst, once)
    assert moves == 0 and twice == once


def test_polish_refuses_to_add_lateness() -> None:
    # customer 2 must be served first (tight window); the shortest order visits it last
    x = np.array([0.0, 1.0, 10.0])
    inst = VRPInstance(
        name="tw",
        coords=np.column_stack([x, np.zeros(3)]),
        durations=np.abs(x[:, None] - x[None]),
        demands=np.zeros(3),
        service_times=np.zeros(3),
        time_windows=np.array([[0, np.inf], [0, np.inf], [0, 10.0]]),
        capacity=np.inf,
    )
    original = [[2, 1]]  # on time, travel 20
    assert evaluate(inst, Solution.from_lists(original)).tw_violations == 0
    assert polish(inst, original) == original


def test_split_giant_tour_roundtrip() -> None:
    routes = [[3, 1], [2], [5, 4]]
    tour = np.array(Solution.from_lists(routes).giant_tour())
    assert split_giant_tour(tour) == routes
    assert split_giant_tour(np.array([0, 0, 1, 0, 0])) == [[1]]


@pytest.mark.parametrize("base", ["nearest_neighbor", "savings"])
def test_ls_solvers_improve_their_base(base: str) -> None:
    inst = random_instance(60, seed=11, capacity=50, num_vehicles=12)
    plain = evaluate(inst, get_solver(base).solve(inst)).travel_time_s
    polished = evaluate(inst, get_solver(f"{base}_ls").solve(inst))
    assert polished.is_complete and polished.capacity_excess == 0
    assert polished.travel_time_s < plain
