import numpy as np
import pytest
import torch

from neural_routing.core import Solution, VRPInstance, evaluate
from neural_routing.ml.env import Batch, RoutingEnv, augment8, generate


def with_windows(batch: Batch, seed: int, tight: bool = True) -> Batch:
    """Random service times and windows, tight enough that many arrivals are late or early."""
    g = torch.Generator().manual_seed(seed)
    B, n1 = batch.size, batch.num_customers + 1
    service = torch.rand(B, n1, generator=g) * 0.05
    service[:, 0] = 0
    start = torch.rand(B, n1, generator=g) * (1.0 if tight else 0.0)
    width = torch.rand(B, n1, generator=g) * (0.5 if tight else 100.0)
    tw = torch.stack([start, start + width], dim=-1)
    tw[:, 0] = torch.tensor([0.0, float("inf")])
    no_window = torch.rand(B, n1, generator=g) < 0.3
    tw[..., 0] = torch.where(no_window, torch.zeros_like(start), tw[..., 0])
    tw[..., 1] = torch.where(no_window, torch.full_like(start, float("inf")), tw[..., 1])
    return Batch(batch.locs, batch.demand, batch.tsp, service, tw)


def rollout(batch: Batch, P: int, seed: int) -> tuple[RoutingEnv, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    env = RoutingEnv(batch, P)
    actions = []
    while not env.all_done:
        probs = (~env.mask()).float()
        a = torch.multinomial(probs.view(batch.size * P, -1), 1, generator=g).view(batch.size, P)
        env.step(a)
        actions.append(a)
    return env, torch.stack(actions, -1)


def to_instance(batch: Batch, b: int) -> VRPInstance:
    xy = batch.locs[b].double().numpy()
    assert batch.service is not None and batch.tw is not None
    return VRPInstance(
        name="t",
        coords=xy,
        durations=np.linalg.norm(xy[:, None] - xy[None], axis=-1),
        demands=batch.demand[b].double().numpy(),
        service_times=batch.service[b].double().numpy(),
        time_windows=batch.tw[b].double().numpy(),
        capacity=np.inf if batch.tsp else 1.0 + 1e-6,
    )


def routes_of(actions: torch.Tensor) -> list[list[int]]:
    routes, cur = [], []
    for a in actions.tolist():
        if a == 0:
            if cur:
                routes.append(cur)
            cur = []
        else:
            cur.append(a)
    return routes


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_env_lateness_matches_the_evaluator(problem: str, seed: int) -> None:
    batch = with_windows(
        generate(6, 12, problem, generator=torch.Generator().manual_seed(seed)), seed
    )
    env, actions = rollout(batch, P=4, seed=seed)
    late_any = False
    for b in range(batch.size):
        inst = to_instance(batch, b)
        for p in range(env.P):
            m = evaluate(inst, Solution.from_lists(routes_of(actions[b, p])))
            assert m.is_complete
            assert m.travel_time_s == pytest.approx(env.length[b, p].item(), rel=1e-5)
            assert m.lateness_s == pytest.approx(env.lateness[b, p].item(), rel=1e-4, abs=1e-5)
            late_any |= m.lateness_s > 0
    assert late_any  # the windows are tight enough for the test to mean something


def test_loose_windows_give_no_lateness_and_no_times_mean_zero() -> None:
    base = generate(4, 10, "cvrp", generator=torch.Generator().manual_seed(5))
    env, _ = rollout(with_windows(base, 5, tight=False), P=3, seed=5)
    assert (env.lateness == 0).all()
    plain, _ = rollout(base, P=3, seed=5)  # a batch without windows never accumulates lateness
    assert (plain.lateness == 0).all()


def test_time_aware_model_learns_signal_and_old_models_still_work() -> None:
    from neural_routing.ml.model import AttentionPolicy, ModelConfig

    small = {"embed_dim": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 64}
    timed = AttentionPolicy(ModelConfig(**small, time_features=True))
    batch = with_windows(generate(4, 10, "cvrp", generator=torch.Generator().manual_seed(3)), 3)
    cost, logp, _ = timed(batch, 10, tw_penalty=10.0)
    assert torch.isfinite(cost).all() and torch.isfinite(logp).all()
    ((cost - cost.mean(1, keepdim=True)) * logp).mean().backward()
    assert timed.embed_node.weight.grad is not None
    assert timed.embed_node.weight.grad[:, 3:].abs().sum() > 0  # time inputs receive gradient
    # a batch without windows is valid input for the timed model, and vice versa
    plain = generate(4, 10, "cvrp")
    assert torch.isfinite(timed(plain, 10)[0]).all()
    untimed = AttentionPolicy(ModelConfig(**small))
    assert untimed.embed_node.in_features == 3
    assert torch.isfinite(untimed(batch, 10)[0]).all()


def test_penalised_cost_adds_lateness() -> None:
    from neural_routing.ml.model import AttentionPolicy, ModelConfig

    torch.manual_seed(0)
    policy = AttentionPolicy(ModelConfig(embed_dim=32, num_heads=4, num_layers=2, ff_dim=64))
    batch = with_windows(generate(4, 10, "cvrp", generator=torch.Generator().manual_seed(4)), 4)
    length, _, a1 = policy(batch, 10, decode="greedy")
    cost, _, a2 = policy(batch, 10, decode="greedy", tw_penalty=10.0)
    assert torch.equal(a1, a2)  # same greedy tours; only the reported cost changes
    env = RoutingEnv(batch, 10)
    for t in range(a1.shape[-1]):
        env.step(a1[..., t])
    torch.testing.assert_close(cost, length + 10.0 * env.lateness)


def test_seconds_per_unit_recovers_the_speed() -> None:
    from neural_routing.ml.generators import normalise
    from neural_routing.solvers.neural import seconds_per_unit

    rng = np.random.default_rng(0)
    km = rng.uniform(0, 5, size=(12, 2))
    inst = VRPInstance(
        name="s", coords=km, durations=100 * np.linalg.norm(km[:, None] - km[None], axis=-1),
        demands=np.zeros(12), service_times=np.zeros(12),
        time_windows=np.tile([0.0, np.inf], (12, 1)), capacity=np.inf,
    )  # fmt: skip
    locs, span = normalise(torch.tensor(km[None]), "all")
    assert seconds_per_unit(inst, locs[0].numpy()) == pytest.approx(100 * span.item(), rel=1e-6)


def test_benchmark_instances_keep_windows_in_the_same_unit_as_travel() -> None:
    from neural_routing.eval.synthetic_benchmark import build

    loose = build("cvrp", 30, 1, "amazon", time_windows=True, tw_tightness=1.0)
    tight = build("cvrp", 30, 1, "amazon", time_windows=True, tw_tightness=0.1)
    plain = build("cvrp", 30, 1, "amazon")
    np.testing.assert_allclose(loose.coords, plain.coords)  # same geometry, only times differ
    assert loose.has_time_windows and not plain.has_time_windows
    w = lambda i: (i.time_windows[:, 1] - i.time_windows[:, 0])[np.isfinite(i.time_windows[:, 1])]  # noqa: E731
    assert np.median(w(tight)) < np.median(w(loose))
    with pytest.raises(ValueError):
        build("cvrp", 30, 1, "uniform", time_windows=True)


def test_time_window_model_trains_and_solves_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from omegaconf import OmegaConf

    from neural_routing.eval.synthetic_benchmark import build
    from neural_routing.ml.train import train
    from neural_routing.solvers import get_solver

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.problem, cfg.num_customers = "cvrp", 8
    cfg.data.update(distribution="amazon", normalisation="customers", time_windows=True)
    cfg.data.tw_tightness = 0.25
    cfg.model = {**cfg.model, "embed_dim": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 64,
                 "norm": "instance", "depot_encoding": "polar", "time_features": True}  # fmt: skip
    cfg.train.update(epochs=1, steps_per_epoch=2, batch_size=4, val_size=8, device="cpu")
    cfg.train.update(run_dir=str(tmp_path / "run"), checkpoint_dir=str(tmp_path / "c"))
    del cfg["hydra"]
    ckpt = train(cfg)
    inst = build("cvrp", 8, 3, "amazon", time_windows=True, tw_tightness=0.25)
    for name in ("attention", "attention_ls"):
        m = evaluate(inst, get_solver(name, checkpoint=ckpt, device="cpu").solve(inst))
        assert m.is_complete and m.capacity_excess == 0


def test_vectorised_candidate_objective_matches_the_evaluator() -> None:
    from neural_routing.solvers.neural import candidate_objectives

    batch = with_windows(generate(1, 12, "cvrp", generator=torch.Generator().manual_seed(9)), 9)
    _, actions = rollout(batch, P=40, seed=9)
    inst = to_instance(batch, 0)
    candidates = actions[0].numpy()
    got = candidate_objectives(inst, candidates, tw_penalty=10.0)
    for c, value in zip(candidates, got, strict=True):
        m = evaluate(inst, Solution.from_lists(routes_of(torch.tensor(c))))
        assert value == pytest.approx(m.travel_time_s + 10.0 * m.lateness_s, rel=1e-9)
    assert (got > np.array([evaluate(inst, Solution.from_lists(routes_of(torch.tensor(c))))
                            .travel_time_s for c in candidates]) - 1e-9).all()  # fmt: skip


def test_augmentation_and_slicing_keep_the_windows() -> None:
    batch = with_windows(generate(3, 10, "cvrp", generator=torch.Generator().manual_seed(7)), 7)
    env, actions = rollout(batch, P=2, seed=7)
    aug = augment8(batch)
    assert aug.tw is not None and aug.tw.shape == (24, 11, 2)
    env8 = RoutingEnv(aug, 2)
    for t in range(actions.shape[-1]):
        env8.step(actions[..., t].repeat(8, 1))
    np.testing.assert_allclose(env8.lateness.view(8, 3, 2), env.lateness.expand(8, 3, 2), rtol=1e-5)
    part = batch.slice(1, 3)
    assert part.size == 2 and part.tw is not None and torch.equal(part.tw, batch.tw[1:3])
