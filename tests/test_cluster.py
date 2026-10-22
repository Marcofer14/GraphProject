import numpy as np
import pytest

from neural_routing.core import Solution, VRPInstance, evaluate
from neural_routing.data.synthetic import random_instance
from neural_routing.solvers import get_solver
from neural_routing.solvers.cluster import capacitated_kmeans, sub_instance


def test_capacitated_kmeans_respects_capacity_and_uses_k_groups() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, size=(120, 2))
    demand = rng.integers(1, 10, size=120).astype(float)
    k = 6
    cap = demand.sum() / k * 1.1  # tight: only 10 % slack
    labels = capacitated_kmeans(xy, demand, k, cap)
    loads = np.bincount(labels, weights=demand, minlength=k)
    assert set(labels.tolist()) <= set(range(k)) and (loads <= cap + 1e-9).all()
    with pytest.raises(ValueError):
        capacitated_kmeans(xy, demand, k, demand.sum() / k * 0.9)


def test_capacitated_kmeans_finds_obvious_clusters() -> None:
    rng = np.random.default_rng(1)
    blobs = [rng.normal(c, 0.2, size=(30, 2)) for c in ((0, 0), (10, 0), (0, 10))]
    labels = capacitated_kmeans(np.vstack(blobs), np.ones(90), 3, 30.0)
    for b in range(3):
        assert len(set(labels[30 * b : 30 * (b + 1)].tolist())) == 1  # each blob is one van


def test_sub_instance_maps_nodes_and_keeps_the_driver_route() -> None:
    inst = random_instance(8, seed=2, capacity=100)
    inst = VRPInstance(**{**inst.__dict__, "reference_routes": ((3, 1, 2), (5, 4, 8, 7, 6))})
    sub = sub_instance(inst, np.array([1, 2, 3]))
    assert sub.num_customers == 3
    np.testing.assert_allclose(sub.durations, inst.durations[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])])
    assert sub.reference_routes == ((3, 1, 2),)  # translated to the group's local ids


@pytest.mark.parametrize("solver", ["cluster_savings_ls"])
def test_cluster_first_solves_a_multi_van_day(solver: str) -> None:
    inst = random_instance(80, seed=3, capacity=60, num_vehicles=8)
    sol = get_solver(solver).solve(inst, time_limit=1.0)
    m = evaluate(inst, sol)
    assert m.is_complete and m.capacity_excess == 0
    assert sol.num_routes <= 8 + 1  # the fleet, give or take a van freed by the polish
    plain = evaluate(inst, get_solver("nearest_neighbor").solve(inst))
    assert m.travel_time_s < plain.travel_time_s


def test_cluster_first_with_the_network(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from omegaconf import OmegaConf

    from neural_routing.ml.train import train

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.problem, cfg.num_customers = "cvrp", 8
    cfg.model = {**cfg.model, "embed_dim": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 64}
    cfg.train.update(epochs=1, steps_per_epoch=2, batch_size=2, val_size=4, device="cpu")
    cfg.train.update(tag="tiny", run_dir=str(tmp_path / "r"), checkpoint_dir=str(tmp_path / "c"))
    del cfg["hydra"]
    ckpt = train(cfg)
    inst = random_instance(30, seed=4, capacity=40, num_vehicles=5)
    sol = get_solver("cluster_attention_ls", checkpoint=ckpt, device="cpu").solve(inst)
    m = evaluate(inst, sol)
    assert m.is_complete and m.capacity_excess == 0
    assert isinstance(sol, Solution)
