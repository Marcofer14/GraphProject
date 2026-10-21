from pathlib import Path

import numpy as np
import pytest
import torch

from neural_routing.core import VRPInstance, evaluate
from neural_routing.ml.env import Batch, RoutingEnv, augment8
from neural_routing.ml.generators import generate_amazon_like, normalise
from neural_routing.ml.model import AttentionPolicy, ModelConfig
from neural_routing.solvers import get_solver
from neural_routing.solvers.neural import _load, to_batch

SMALL = {"embed_dim": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 64, "norm": "instance"}


def far_depot_km() -> torch.Tensor:
    """Depot 20 km away from four customers on a 2 x 1 km rectangle."""
    return torch.tensor([[[20.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]])


@pytest.mark.parametrize("mode", ["all", "customers"])
def test_normalisation_is_a_pure_rescaling(mode: str) -> None:
    km = far_depot_km()
    locs, span = normalise(km, mode)
    # pairwise distances are the km distances divided by one scalar
    np.testing.assert_allclose(torch.cdist(locs, locs) * span, torch.cdist(km, km), atol=1e-5)


def test_customer_mode_fills_the_square_and_lets_the_depot_out() -> None:
    locs, span = normalise(far_depot_km(), "customers")
    customers = locs[0, 1:]
    assert span.item() == pytest.approx(2.0)  # the customers' widest side
    assert customers.min() == 0 and customers.max() == pytest.approx(1.0)
    assert locs[0, 0, 0] == pytest.approx(10.0)  # depot 20 km away = 10 customer-box widths

    squashed, _ = normalise(far_depot_km(), "all")
    assert squashed[0, 1:].max() < 0.11  # the old behaviour: customers in a corner


def test_unknown_normalisation() -> None:
    with pytest.raises(ValueError):
        normalise(far_depot_km(), "zscore")


def test_polar_depot_features_are_bounded() -> None:
    policy = AttentionPolicy(ModelConfig(**SMALL, depot_encoding="polar"))
    locs, _ = normalise(far_depot_km() * 1000, "customers")  # absurdly far depot
    feats = policy.depot_features(Batch(locs, torch.zeros(1, 5), tsp=True))
    assert feats.shape == (1, 1, 3) and torch.isfinite(feats).all()
    assert feats[0, 0, :2].norm().item() == pytest.approx(1.0)  # a unit direction
    assert feats[0, 0, 2].item() < 10  # log1p of ~10,000


def test_polar_model_runs_and_augmentation_still_preserves_lengths() -> None:
    batch, _ = generate_amazon_like(4, 12, "cvrp", generator=torch.Generator().manual_seed(0),
                                    normalisation="customers")  # fmt: skip
    assert (batch.locs[:, 0].abs() > 1).any()  # the depot really is outside the square
    policy = AttentionPolicy(ModelConfig(**SMALL, depot_encoding="polar"))
    length, logp, actions = policy(batch, 12)
    assert torch.isfinite(length).all() and torch.isfinite(logp).all()
    env = RoutingEnv(augment8(batch), 12)
    for t in range(actions.shape[-1]):
        env.step(actions[..., t].repeat(8, 1))
    np.testing.assert_allclose(env.length.view(8, 4, 12), length.expand(8, 4, 12), rtol=1e-4)


def latlng_square() -> VRPInstance:
    """Depot plus 1 km east and 1 km north of it, around latitude 47.6 (Seattle)."""
    lat0 = 47.6
    dlat = 1 / 110.574
    dlng = 1 / (111.32 * np.cos(np.radians(lat0)))
    coords = np.array([[lat0, -122.3], [lat0, -122.3 + dlng], [lat0 + dlat, -122.3]])
    D = np.full((3, 3), 60.0)
    np.fill_diagonal(D, 0)
    return VRPInstance(
        name="sq", coords=coords, durations=D, demands=np.zeros(3), service_times=np.zeros(3),
        time_windows=np.tile([0.0, np.inf], (3, 1)), capacity=np.inf,
        meta={"coord_system": "latlng"},
    )  # fmt: skip


def test_latlng_is_projected_before_normalising() -> None:
    locs = to_batch(latlng_square(), "cpu").locs[0]
    east, north = (locs[1] - locs[0]).norm(), (locs[2] - locs[0]).norm()
    assert east.item() == pytest.approx(north.item(), rel=1e-3)  # degrees would differ by ~33 %


def _train_tiny(tmp_path: Path, **data: str) -> Path:
    from omegaconf import OmegaConf

    from neural_routing.ml.train import train

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.problem, cfg.num_customers = "cvrp", 8
    cfg.data.update(distribution="amazon", **data)
    cfg.model = {**cfg.model, **SMALL, "depot_encoding": "polar"}
    cfg.train.update(epochs=1, steps_per_epoch=2, batch_size=4, val_size=8, device="cpu")
    cfg.train.update(run_dir=str(tmp_path / "run"), checkpoint_dir=str(tmp_path / "ckpt"))
    del cfg["hydra"]
    return train(cfg)


def test_solver_uses_the_checkpoint_normalisation(tmp_path: Path) -> None:
    from neural_routing.data.synthetic import random_instance

    ckpt = _train_tiny(tmp_path, normalisation="customers")
    assert _load(str(ckpt), "cpu")[1] == "customers"
    inst = random_instance(8, seed=2, capacity=40)
    for name in ("attention", "attention_ls"):
        m = evaluate(inst, get_solver(name, checkpoint=ckpt, device="cpu").solve(inst))
        assert m.is_complete and m.capacity_excess == 0


def test_old_checkpoints_default_to_all(tmp_path: Path) -> None:
    ckpt = _train_tiny(tmp_path, normalisation="customers")
    raw = torch.load(ckpt, weights_only=False)
    del raw["train_config"]["data"]  # what a checkpoint from before this change looks like
    old = tmp_path / "old.pt"
    torch.save(raw, old)
    assert _load(str(old), "cpu")[1] == "all"
