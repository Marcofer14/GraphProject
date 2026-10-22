import numpy as np
import pytest
import torch

from neural_routing.core import Solution, VRPInstance, evaluate
from neural_routing.ml.env import RoutingEnv, augment8, generate
from neural_routing.ml.model import AttentionPolicy, ModelConfig

SMALL = ModelConfig(embed_dim=32, num_heads=4, num_layers=2, ff_dim=64)


def random_rollout(problem: str, n: int = 12, B: int = 4, P: int = 3, seed: int = 0):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    batch = generate(B, n, problem, generator=g)
    env = RoutingEnv(batch, P)
    actions = []
    while not env.all_done:
        m = env.mask()
        assert (~m).any(-1).all(), "every trajectory must have a feasible move"
        probs = (~m).float()
        a = torch.multinomial(probs.view(B * P, -1), 1, generator=g).view(B, P)
        env.step(a)
        actions.append(a)
    return batch, env, torch.stack(actions, -1)


def to_instance(batch, b: int) -> VRPInstance:  # type: ignore[no-untyped-def]
    xy = batch.locs[b].double().numpy()
    n = xy.shape[0]
    return VRPInstance(
        name="t",
        coords=xy,
        durations=np.linalg.norm(xy[:, None] - xy[None], axis=-1),
        demands=batch.demand[b].double().numpy(),
        service_times=np.zeros(n),
        time_windows=np.tile([0.0, np.inf], (n, 1)),
        capacity=np.inf if batch.tsp else 1.0 + 1e-6,
    )


def routes_from(actions: torch.Tensor) -> list[list[int]]:
    routes, cur = [], []
    for a in actions.tolist():
        if a == 0:
            routes.append(cur) if cur else None
            cur = []
        else:
            cur.append(a)
    return routes


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_env_matches_evaluator(problem: str) -> None:
    batch, env, actions = random_rollout(problem)
    for b in range(batch.size):
        inst = to_instance(batch, b)
        for p in range(env.P):
            routes = routes_from(actions[b, p])
            m = evaluate(inst, Solution.from_lists(routes))
            assert m.is_complete and m.capacity_excess == 0
            assert m.travel_time_s == pytest.approx(env.length[b, p].item(), rel=1e-5)
            if problem == "tsp":
                assert len(routes) == 1


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_policy_forward_backward(problem: str) -> None:
    torch.manual_seed(0)
    policy = AttentionPolicy(SMALL)
    batch = generate(3, 8, problem)
    length, logp, actions = policy(batch, num_starts=8, decode="sample")
    assert length.shape == logp.shape == (3, 8)
    assert (actions[:, :, 0] == torch.arange(1, 9)).all()  # POMO forced starts
    assert torch.isfinite(logp).all() and (logp <= 0).all()
    ((length - length.mean(1, keepdim=True)) * logp).mean().backward()
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)


def test_greedy_is_deterministic_and_rollout_mode() -> None:
    policy = AttentionPolicy(SMALL).eval()
    batch = generate(2, 10, "cvrp", generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        a = policy(batch, 1, decode="greedy", forced_starts=False)[0]
        b = policy(batch, 1, decode="greedy", forced_starts=False)[0]
    assert torch.equal(a, b)


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_train_and_neural_solver_end_to_end(problem: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from omegaconf import OmegaConf

    from neural_routing.data.synthetic import random_instance
    from neural_routing.ml.env import default_capacity
    from neural_routing.ml.train import train
    from neural_routing.solvers import get_solver

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.problem, cfg.num_customers = problem, 8
    cfg.model = {**cfg.model, **vars(SMALL)}
    cfg.train.update(epochs=2, steps_per_epoch=3, batch_size=4, val_size=8, device="cpu")
    cfg.train.update(run_dir=str(tmp_path / "run"), checkpoint_dir=str(tmp_path / "ckpt"))
    del cfg["hydra"]
    ckpt = train(cfg)
    assert ckpt.exists() and (tmp_path / "run" / "history.json").exists()

    cap = np.inf if problem == "tsp" else default_capacity(8)
    inst = random_instance(8, seed=3, capacity=cap)
    for name in ["attention", "attention_greedy", "attention_ls"]:
        sol = get_solver(name, checkpoint=ckpt, device="cpu").solve(inst)
        m = evaluate(inst, sol)
        assert m.is_complete and m.capacity_excess == 0


def test_training_resumes_from_last_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from omegaconf import OmegaConf

    from neural_routing.ml.train import train

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.num_customers = 6
    cfg.model = {**cfg.model, **vars(SMALL)}
    cfg.train.update(epochs=1, steps_per_epoch=2, batch_size=4, val_size=4, device="cpu")
    cfg.train.update(run_dir=str(tmp_path / "run1"), checkpoint_dir=str(tmp_path / "ckpt"))
    del cfg["hydra"]
    train(cfg)
    cfg.train.update(epochs=3, run_dir=str(tmp_path / "run2"))
    train(cfg)
    history = json.loads((tmp_path / "run2" / "history.json").read_text())
    assert [h["epoch"] for h in history] == [0, 1, 2]
    assert history[-1]["step"] == 6


def test_early_stopping_counts_only_significant_improvements() -> None:
    from neural_routing.ml.train import EarlyStopping

    es = EarlyStopping(patience=2, min_rel_delta=0.01)
    assert not es.update(10.0)  # first value sets the reference
    assert not es.update(9.95)  # 0.5 % better: below the 1 % threshold, counts as stale
    assert not es.update(9.80)  # 2 % better: resets the counter
    assert not es.update(9.79)
    assert es.update(9.78)  # second stale epoch in a row -> stop
    assert not EarlyStopping(patience=None).update(1.0)


def test_unavoidable_cost_is_the_nearest_round_trip() -> None:
    from neural_routing.ml.env import Batch
    from neural_routing.ml.train import unavoidable_cost

    # depot at x=10; nearest customer at x=1 (distance 9) and x=2 (distance 8)
    locs = torch.tensor(
        [[[10.0, 0.0], [0.0, 0.0], [1.0, 0.0]], [[10.0, 0.0], [2.0, 0.0], [0.0, 0.0]]]
    )
    assert unavoidable_cost(Batch(locs, torch.zeros(2, 3), tsp=True)) == pytest.approx(17.0)
    # it is a true lower bound: every tour of a random instance is at least that long
    batch = generate(64, 10, "cvrp", generator=torch.Generator().manual_seed(0))
    length, _, _ = AttentionPolicy(SMALL)(batch, 10, decode="sample")
    gaps = (batch.locs[:, 1:] - batch.locs[:, :1]).norm(dim=-1).min(dim=1).values
    assert (length >= 2 * gaps[:, None] - 1e-5).all()


def test_training_stops_early_and_stays_stopped_on_resume(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from omegaconf import OmegaConf

    from neural_routing.ml.train import train

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.num_customers = 6
    # instance norm keeps no running statistics, so with lr = 0 the policy is truly frozen
    # and validation never improves
    cfg.model = {**cfg.model, **vars(SMALL), "norm": "instance"}
    cfg.train.update(epochs=20, steps_per_epoch=1, batch_size=2, val_size=4, device="cpu", lr=0.0)
    cfg.train.update(run_dir=str(tmp_path / "run"), checkpoint_dir=str(tmp_path / "ckpt"))
    cfg.train.early_stopping = {"patience": 2, "min_rel_delta": 0.0}
    del cfg["hydra"]
    train(cfg)
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert len(history) == 3  # best at epoch 0, then 2 stale epochs

    cfg.train.run_dir = str(tmp_path / "run2")
    train(cfg)  # resuming must not train past the stopping point
    assert not (tmp_path / "run2" / "history.json").exists()


def test_logging_failure_does_not_kill_training(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter

    from neural_routing.ml.train import train

    def locked(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise PermissionError("file locked by a sync client")

    monkeypatch.setattr(SummaryWriter, "add_scalar", locked)
    cfg = OmegaConf.load("configs/base.yaml")
    cfg.num_customers = 6
    cfg.model = {**cfg.model, **vars(SMALL)}
    cfg.train.update(epochs=2, steps_per_epoch=2, batch_size=2, val_size=4, device="cpu")
    cfg.train.update(log_every=1, run_dir=str(tmp_path / "run"), checkpoint_dir=str(tmp_path / "c"))
    del cfg["hydra"]
    assert train(cfg).exists()  # training finished despite every log write failing


def test_checkpoint_save_retries_transient_lock(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import neural_routing.ml.train as tr

    monkeypatch.setattr(tr.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky() -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")

    tr._retry_io(flaky, "x")
    assert calls["n"] == 3
    with pytest.raises(PermissionError):
        tr._retry_io(lambda: (_ for _ in ()).throw(PermissionError("always")), "x", attempts=2)


def test_init_from_starts_a_new_stage_from_other_weights(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from omegaconf import OmegaConf

    from neural_routing.ml.train import train

    def cfg_for(n: int, tag: str):  # type: ignore[no-untyped-def]
        cfg = OmegaConf.load("configs/base.yaml")
        cfg.num_customers = n
        # instance norm: no running statistics, so lr = 0 really freezes every saved tensor
        cfg.model = {**cfg.model, **vars(SMALL), "norm": "instance"}
        cfg.train.update(epochs=1, steps_per_epoch=2, batch_size=2, val_size=4, device="cpu")
        cfg.train.update(tag=tag, run_dir=str(tmp_path / tag), checkpoint_dir=str(tmp_path / "c"))
        del cfg["hydra"]
        return cfg

    stage1 = train(cfg_for(6, "stage1"))
    w1 = torch.load(stage1, weights_only=False)["state_dict"]

    cfg2 = cfg_for(9, "stage2")  # a different size: the policy is size-agnostic
    cfg2.train.lr = 0.0  # frozen, so the stage-2 weights must equal stage 1's exactly
    stage2 = train(cfg2)
    w2 = torch.load(stage2, weights_only=False)["state_dict"]
    assert any(not torch.equal(w1[k], w2[k]) for k in w1)  # not initialised: random weights

    cfg2b = cfg_for(9, "stage2b")
    cfg2b.train.update(lr=0.0, init_from=str(stage1))
    w2b = torch.load(train(cfg2b), weights_only=False)["state_dict"]
    assert all(torch.equal(w1[k], w2b[k]) for k in w1)
    history = json.loads((tmp_path / "stage2b" / "history.json").read_text())
    assert [h["epoch"] for h in history] == [0]  # fresh history, not a resume

    # a checkpoint saved before a (default-valued) model option existed is still accepted
    old = torch.load(stage1, weights_only=False)
    del old["model_config"]["time_features"]
    old_path = tmp_path / "old_stage1.pt"
    torch.save(old, old_path)
    cfg2c = cfg_for(9, "stage2c")
    cfg2c.train.init_from = str(old_path)
    assert train(cfg2c).exists()

    bad = cfg_for(9, "bad")
    bad.model.embed_dim = 64
    bad.model.num_heads = 4
    bad.train.init_from = str(stage1)
    with pytest.raises(ValueError, match="model config"):
        train(bad)


def test_augment8_preserves_tour_length() -> None:
    batch, env, actions = random_rollout("tsp", B=2, P=1)
    aug = augment8(batch)
    env2 = RoutingEnv(aug, 1)
    for t in range(actions.shape[-1]):
        env2.step(actions[..., t].repeat(8, 1))
    np.testing.assert_allclose(
        env2.length.view(8, 2), env.length.view(1, 2).expand(8, 2), rtol=1e-5
    )
