import itertools

import numpy as np
import pytest

from neural_routing.core import evaluate
from neural_routing.data.synthetic import random_instance
from neural_routing.solvers import available_solvers, get_solver

SOLVERS = [s for s in available_solvers() if s != "reference" and not s.startswith("attention")]


@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("tw", [False, True])
def test_solver_visits_every_customer_once(name: str, tw: bool) -> None:
    inst = random_instance(30, seed=1, capacity=40, num_vehicles=4, with_time_windows=tw)
    sol = get_solver(name).solve(inst, time_limit=0.5)
    m = evaluate(inst, sol)
    assert m.is_complete
    assert m.capacity_excess == 0
    assert sol.solver == name


def brute_force_tsp(D: np.ndarray) -> float:
    n = D.shape[0]
    best = np.inf
    for perm in itertools.permutations(range(1, n)):
        tour = (0, *perm, 0)
        best = min(best, sum(D[a, b] for a, b in itertools.pairwise(tour)))
    return float(best)


@pytest.mark.parametrize("name", ["ortools", "pyvrp"])
def test_metaheuristics_optimal_on_small_tsp(name: str) -> None:
    inst = random_instance(7, seed=3, capacity=np.inf)
    sol = get_solver(name).solve(inst, time_limit=0.5)
    m = evaluate(inst, sol)
    assert m.num_routes == 1
    # solvers work on rounded integer times, so allow rounding slack
    assert m.travel_time_s <= brute_force_tsp(inst.durations) + inst.num_nodes


def test_metaheuristics_beat_construction() -> None:
    inst = random_instance(60, seed=7, capacity=60, num_vehicles=8)
    cost = {s: evaluate(inst, get_solver(s).solve(inst, 1.0)).travel_time_s for s in SOLVERS}
    assert cost["pyvrp"] <= cost["savings"] <= cost["nearest_neighbor"]
    assert cost["ortools"] <= cost["nearest_neighbor"]


def test_savings_respects_single_vehicle() -> None:
    # neighbour-restricted savings used to leave fragments on uncapacitated single-van instances
    inst = random_instance(150, seed=5, capacity=np.inf)
    sol = get_solver("savings", k=5).solve(inst)
    assert sol.num_routes == 1 and evaluate(inst, sol).is_complete


def test_unknown_solver() -> None:
    with pytest.raises(KeyError):
        get_solver("does-not-exist")
