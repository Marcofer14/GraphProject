import math
from pathlib import Path

import pytest
import torch

from neural_routing.data.calibration import DEFAULT_PATH, GeneratorParams, project_km
from neural_routing.ml.generators import generate_amazon_like, load_params, sample_real


def gen(seed: int = 0, **kw):  # type: ignore[no-untyped-def]
    return generate_amazon_like(256, 50, generator=torch.Generator().manual_seed(seed), **kw)


def test_shapes_ranges_and_reproducibility() -> None:
    batch, times = gen()
    assert batch.locs.shape == (256, 51, 2) and batch.demand.shape == (256, 51)
    assert batch.locs.min() >= 0 and batch.locs.max() <= 1 + 1e-6
    assert (batch.demand[:, 0] == 0).all() and (batch.demand <= 1).all()
    assert times.service.shape == (256, 51) and times.tw.shape == (256, 51, 2)
    again, _ = gen()
    assert torch.equal(batch.locs, again.locs) and torch.equal(batch.demand, again.demand)


def test_geometry_is_calibrated_far_station_clustered_stops() -> None:
    batch, _ = gen()
    customers = batch.locs[:, 1:]
    centroid = customers.mean(1)
    depot_dist = (batch.locs[:, 0] - centroid).norm(dim=-1)
    spread = (customers - centroid[:, None]).norm(dim=-1).mean(1)
    # the real station is ~17 km from routes ~3.5 km wide: far relative to the customers' spread
    assert (depot_dist / spread).median() > 3


def test_real_demands_are_tiny_unless_capacity_is_forced() -> None:
    real, _ = gen()
    assert real.demand[:, 1:].sum(1).median() < 0.5  # 50 real stops fill well under one van
    forced, _ = gen(stops_per_vehicle=10)
    vans = forced.demand[:, 1:].sum(1)
    assert 3 < vans.median() < 7  # ~50 / 10 vans' worth of load


def test_time_windows_follow_the_data() -> None:
    p = load_params()
    _, times = generate_amazon_like(2000, 50, generator=torch.Generator().manual_seed(1))
    windowed = torch.isfinite(times.tw[:, 1:, 1])
    assert abs(windowed.float().mean().item() - p.tw_probability) < 0.01
    assert (times.tw[..., 0] >= 0).all() and (times.tw[..., 1] >= times.tw[..., 0]).all()
    assert torch.isinf(times.tw[:, 0, 1]).all() and (times.service[:, 0] == 0).all()
    # ~73 s median service at ~35 km/h is a small fraction of a normalised unit
    assert (times.service[:, 1:] > 0).all()


def test_tsp_has_no_demand() -> None:
    batch, _ = generate_amazon_like(4, 10, problem="tsp")
    assert batch.tsp and (batch.demand == 0).all()


def test_params_roundtrip(tmp_path: Path) -> None:
    params = GeneratorParams.load(DEFAULT_PATH)
    params.save(tmp_path / "p.json")
    assert GeneratorParams.load(tmp_path / "p.json") == params
    assert 10 < params.depot_distance_km.median < 30  # sanity: stations are ~17 km away
    assert 80 < params.travel_s_per_km < 130


def test_sample_real_on_fixture(amazon_raw: Path) -> None:
    from neural_routing.data.amazon import AmazonDataset
    from neural_routing.data.amazon_etl import run

    out = run("sample", raw_root=amazon_raw / "raw", out_root=amazon_raw / "processed")
    ds = AmazonDataset("sample", root=out.parent)
    stops = project_km(ds.stops)
    batch, times = sample_real(stops, ds.routes["capacity_cm3"], count=5, num_customers=3, seed=0)
    assert batch.locs.shape == (5, 4, 2)
    assert batch.locs.min() >= 0 and batch.locs.max() <= 1 + 1e-6
    assert torch.allclose(batch.demand[:, 1:], torch.full((5, 3), 1000 / 1_000_000))
    assert torch.isfinite(times.tw[..., 1]).any()  # stop A1/B1 carries a window


def test_sample_real_takes_exactly_n_stops_across_zones() -> None:
    import pandas as pd

    # three zones of 4 stops on a line: A (x~0), B (x~1), C (x~10); C is far from A and B
    rows = [
        {"route_id": "R", "zone": z, "x": cx + 0.01 * k, "y": 0.0, "volume_cm3": 1.0,
         "service_s": 60.0, "tw_start_s": float("nan"), "tw_end_s": float("nan")}
        for z, cx in (("A", 0.0), ("B", 1.0), ("C", 10.0))
        for k in range(4)
    ]  # fmt: skip
    stops = pd.DataFrame(rows)
    # 6 stops span two zones (4 + 2): the old code counted zones instead of stops here
    batch, _ = sample_real(stops, pd.Series({"R": 100.0}), count=20, num_customers=6, seed=1)
    assert batch.locs.shape == (20, 7, 2)
    assert torch.allclose(batch.demand[:, 1:], torch.full((20, 6), 0.01))


def test_training_on_amazon_distribution(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from neural_routing.ml.train import train

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.problem, cfg.num_customers = "cvrp", 8
    cfg.data.distribution = "amazon"
    cfg.model = {**cfg.model, "embed_dim": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 64}
    cfg.train.update(epochs=1, steps_per_epoch=2, batch_size=4, val_size=8, device="cpu")
    cfg.train.update(run_dir=str(tmp_path / "run"), checkpoint_dir=str(tmp_path / "ckpt"))
    del cfg["hydra"]
    ckpt = train(cfg)
    assert ckpt.name == "cvrp8_pomo_amazon.pt" and ckpt.exists()
    assert math.isfinite(torch.load(ckpt, weights_only=False)["val_length"])


def test_run_folder_follows_the_experiment_name() -> None:
    from hydra import compose, initialize_config_dir

    from neural_routing.ml.train import CONFIG_DIR

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        names = {
            c: compose(config_name=c).train.run_dir.split("/")[1]
            for c in ("cvrp50", "cvrp50_amazon", "cvrp50_amazon_all")
        }
    assert names == {
        "cvrp50": "cvrp50_pomo",
        "cvrp50_amazon": "cvrp50_pomo_amazon",
        "cvrp50_amazon_all": "cvrp50_pomo_amazon_all",
    }


@pytest.mark.parametrize("bad", ["gaussian"])
def test_unknown_distribution(bad: str, tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from neural_routing.ml.train import make_sampler

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.data.distribution = bad
    with pytest.raises(ValueError):
        make_sampler(cfg, torch.device("cpu"))
