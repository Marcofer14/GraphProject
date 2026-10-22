import itertools

import numpy as np
import pytest

from neural_routing.core import evaluate
from neural_routing.data.synthetic import random_instance
from neural_routing.solvers import available_solvers, get_solver

# neural solvers need a trained checkpoint (absent in CI); tiny models test them elsewhere
SOLVERS = [s for s in available_solvers() if s != "reference" and "attention" not in s]


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


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ("savings", ("savings", 1.0)),
        ("pyvrp:10", ("pyvrp", 10.0)),
        ("attention@checkpoints/cvrp50_pomo.pt", ("attention", 1.0)),
        ("attention_ls@runs/a_b.pt:2", ("attention_ls", 2.0)),
    ],
)
def test_parse_config(cfg: str, expected: tuple[str, float]) -> None:
    from neural_routing.eval.benchmark import parse_config

    assert parse_config(cfg) == expected


def test_amazon_like_benchmark_instance() -> None:
    from neural_routing.eval.synthetic_benchmark import build

    inst = build("cvrp", 20, seed=3, distribution="amazon")
    assert inst.num_customers == 20 and inst.capacity == 1.0
    m = evaluate(inst, get_solver("savings").solve(inst))
    assert m.is_complete and m.capacity_excess == 0


def test_controllable_gap_ignores_the_fixed_round_trip() -> None:
    import pandas as pd

    from neural_routing.eval.benchmark import add_gaps, neural_kwargs, unavoidable_s

    inst = random_instance(10, seed=0, capacity=np.inf)
    D = inst.durations
    assert unavoidable_s(inst) == pytest.approx(D[0, 1:].min() + D[1:, 0].min())
    # fixed 80 of 100 vs 110: +10 % on the tour, +50 % on the part a solver controls
    df = pd.DataFrame(
        {"instance": ["a", "a"], "config": ["x", "y"], "travel_time_s": [100.0, 110.0],
         "fixed_s": [80.0, 80.0], "is_complete": [True, True]}
    )  # fmt: skip
    out = add_gaps(df)
    assert out["gap_pct"].tolist() == pytest.approx([0.0, 10.0])
    assert out["gap_ctrl_pct"].tolist() == pytest.approx([0.0, 50.0])
    kw = neural_kwargs(["savings", "attention_ls@a.pt", "attention"], default_checkpoint="d.pt")
    assert kw == {"attention_ls@a.pt": {"checkpoint": "a.pt"}, "attention": {"checkpoint": "d.pt"}}


def test_unknown_solver() -> None:
    with pytest.raises(KeyError):
        get_solver("does-not-exist")
