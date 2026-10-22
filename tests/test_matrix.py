import numpy as np
import pytest
import torch

from neural_routing.core import Solution, VRPInstance, evaluate
from neural_routing.ml.env import Batch, RoutingEnv, augment8, generate
from neural_routing.ml.model import AttentionPolicy, ModelConfig

SMALL = {"embed_dim": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 64, "norm": "instance"}


def with_matrix(batch: Batch, seed: int) -> Batch:
    """Asymmetric 'road' times: straight line x detour in [1, 2.5] plus a fixed cost per leg."""
    g = torch.Generator().manual_seed(seed)
    euclid = torch.cdist(batch.locs, batch.locs)
    detour = 1 + 1.5 * torch.rand(euclid.shape, generator=g)
    dist = euclid * detour + 0.05
    dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[1])] = 0
    return Batch(batch.locs, batch.demand, batch.tsp, batch.service, batch.tw, dist)


def rollout(batch: Batch, P: int, seed: int) -> tuple[RoutingEnv, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    env = RoutingEnv(batch, P)
    actions = []
    while not env.all_done:
        a = torch.multinomial((~env.mask()).float().view(batch.size * P, -1), 1, generator=g)
        a = a.view(batch.size, P)
        env.step(a)
        actions.append(a)
    return env, torch.stack(actions, -1)


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
def test_env_charges_the_matrix_exactly_like_the_evaluator(problem: str) -> None:
    batch = with_matrix(generate(5, 10, problem, generator=torch.Generator().manual_seed(1)), 1)
    assert batch.dist is not None
    env, actions = rollout(batch, P=3, seed=1)
    for b in range(batch.size):
        n1 = batch.num_customers + 1
        inst = VRPInstance(
            name="m", coords=batch.locs[b].double().numpy(),
            durations=batch.dist[b].double().numpy(), demands=batch.demand[b].double().numpy(),
            service_times=np.zeros(n1), time_windows=np.tile([0.0, np.inf], (n1, 1)),
            capacity=np.inf if batch.tsp else 1.0 + 1e-6,
        )  # fmt: skip
        for p in range(env.P):
            m = evaluate(inst, Solution.from_lists(routes_of(actions[b, p])))
            assert m.is_complete
            assert m.travel_time_s == pytest.approx(env.length[b, p].item(), rel=1e-5)


def test_augmentation_keeps_the_matrix() -> None:
    batch = with_matrix(generate(2, 8, "cvrp", generator=torch.Generator().manual_seed(2)), 2)
    env, actions = rollout(batch, P=2, seed=2)
    env8 = RoutingEnv(augment8(batch), 2)
    for t in range(actions.shape[-1]):
        env8.step(actions[..., t].repeat(8, 1))
    np.testing.assert_allclose(env8.length.view(8, 2, 2), env.length.expand(8, 2, 2), rtol=1e-5)


def test_fresh_matrix_model_behaves_exactly_like_the_coordinate_model() -> None:
    torch.manual_seed(0)
    plain = AttentionPolicy(ModelConfig(**SMALL))
    matrix = AttentionPolicy(ModelConfig(**SMALL, matrix_bias=True))
    missing, unexpected = matrix.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(("edge_bias", "step_bias")) for k in missing)
    batch = with_matrix(generate(3, 10, "cvrp", generator=torch.Generator().manual_seed(3)), 3)
    a = plain.eval()(batch, 10, decode="greedy")
    b = matrix.eval()(batch, 10, decode="greedy")
    assert torch.equal(a[2], b[2])  # same tours: zero-initialised biases are a no-op
    torch.testing.assert_close(a[0], b[0])


def test_matrix_model_warm_starts_from_a_coordinate_model_and_solves(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from omegaconf import OmegaConf

    from neural_routing.data.synthetic import random_instance
    from neural_routing.ml.train import train
    from neural_routing.solvers import get_solver

    def cfg_for(tag: str, matrix: bool):  # type: ignore[no-untyped-def]
        cfg = OmegaConf.load("configs/base.yaml")
        cfg.problem, cfg.num_customers = "cvrp", 8
        cfg.model = {**cfg.model, **SMALL, "matrix_bias": matrix}
        cfg.train.update(epochs=1, steps_per_epoch=2, batch_size=2, val_size=4, device="cpu")
        cfg.train.update(tag=tag, run_dir=str(tmp_path / tag), checkpoint_dir=str(tmp_path / "c"))
        del cfg["hydra"]
        return cfg

    coord = train(cfg_for("coord", matrix=False))
    seeded = cfg_for("matrix", matrix=True)
    seeded.train.init_from = str(coord)
    ckpt = train(seeded)
    state = torch.load(ckpt, weights_only=False)["state_dict"]
    assert "edge_bias.2.weight" in state
    # an instance with a real (here: random asymmetric) matrix goes through the matrix path
    inst = random_instance(8, seed=1, capacity=40)
    rng = np.random.default_rng(1)
    D = inst.durations * rng.uniform(1.0, 2.5, inst.durations.shape)
    np.fill_diagonal(D, 0)
    inst = VRPInstance(**{**inst.__dict__, "durations": D})
    for name in ("attention", "attention_ls"):
        m = evaluate(inst, get_solver(name, checkpoint=ckpt, device="cpu").solve(inst))
        assert m.is_complete and m.capacity_excess == 0


def test_matrix_biases_learn_and_work_without_a_matrix() -> None:
    torch.manual_seed(0)
    policy = AttentionPolicy(ModelConfig(**SMALL, matrix_bias=True))
    batch = with_matrix(generate(4, 10, "cvrp", generator=torch.Generator().manual_seed(4)), 4)
    cost, logp, _ = policy(batch, 10)
    ((cost - cost.mean(1, keepdim=True)) * logp).mean().backward()
    # the last (zero) layer gets gradient first; that is what lets the biases grow from no-op
    assert policy.edge_bias[2].weight.grad.abs().sum() > 0
    assert policy.step_bias[2].weight.grad.abs().sum() > 0
    plain_batch = generate(2, 10, "cvrp")  # no matrix: falls back to straight-line distances
    assert torch.isfinite(policy(plain_batch, 10)[0]).all()
