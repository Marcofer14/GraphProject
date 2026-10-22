from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neural_routing.api.app import create_app
from neural_routing.data.amazon_etl import run


@pytest.fixture
def client(amazon_raw: Path, tmp_path: Path) -> TestClient:
    run("sample", raw_root=amazon_raw / "raw", out_root=amazon_raw / "processed")
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    return TestClient(create_app(data_root=amazon_raw / "processed", checkpoint_dir=ckpt_dir))


def tiny_model(ckpt_dir: Path) -> str:
    from omegaconf import OmegaConf

    from neural_routing.ml.train import train

    cfg = OmegaConf.load("configs/base.yaml")
    cfg.problem, cfg.num_customers = "cvrp", 6
    cfg.model = {**cfg.model, "embed_dim": 32, "num_heads": 4, "num_layers": 2, "ff_dim": 64}
    cfg.train.update(epochs=1, steps_per_epoch=2, batch_size=2, val_size=4, device="cpu")
    cfg.train.update(tag="tiny", run_dir=str(ckpt_dir / "run"), checkpoint_dir=str(ckpt_dir))
    del cfg["hydra"]
    train(cfg)
    return "tiny"


def test_health_solvers_and_instances(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    solvers = {s["name"]: s["kind"] for s in client.get("/solvers").json()}
    assert solvers["pyvrp"] == "classic" and solvers["attention_ls"] == "neural"
    assert solvers["reference"] == "reference"
    routes = client.get("/instances", params={"split": "sample"}).json()
    assert {r["route_id"] for r in routes} == {"R1", "R2"}
    assert routes[0]["num_stops"] == 3


def test_instance_detail_has_coordinates_windows_and_driver(client: TestClient) -> None:
    detail = client.get("/instances/sample/R1").json()
    assert detail["stops"][0]["node"] == 0 and detail["stops"][0]["stop_id"] == "ST"
    a1 = next(s for s in detail["stops"] if s["stop_id"] == "A1")
    assert (a1["tw_start_s"], a1["tw_end_s"]) == (1800.0, 7200.0)  # 15:30-17:00, departure 15:00
    a0 = next(s for s in detail["stops"] if s["stop_id"] == "A0")
    assert a0["tw_end_s"] is None
    assert detail["driver_route"] == [3, 2, 1]


def test_solve_and_compare(client: TestClient) -> None:
    sol = client.post("/solve", json={"split": "sample", "route_id": "R1", "solver": "savings"})
    assert sol.status_code == 200
    body = sol.json()
    assert sorted(v for r in body["routes"] for v in r) == [1, 2, 3]
    assert body["paths"][0][0] == body["paths"][0][-1]  # starts and ends at the station
    assert len(body["paths"][0][0]) == 2  # [lng, lat]
    cmp = client.post(
        "/compare",
        json={"split": "sample", "route_id": "R1", "solvers": ["reference", "savings_ls", "pyvrp"],
              "time_limit": 0.5},
    ).json()  # fmt: skip
    gaps = {s["solver"]: s["gap_pct"] for s in cmp["solutions"]}
    assert min(gaps.values()) == 0 and gaps[cmp["best"]] == 0
    assert gaps["reference"] >= gaps["pyvrp"]


def test_errors_are_clean(client: TestClient) -> None:
    assert (
        client.post(
            "/solve", json={"split": "sample", "route_id": "nope", "solver": "savings"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/solve", json={"split": "sample", "route_id": "R1", "solver": "magic"}
        ).status_code
        == 400
    )
    too_long = {"split": "sample", "route_id": "R1", "solver": "pyvrp", "time_limit": 999}
    assert client.post("/solve", json=too_long).status_code == 422
    assert client.get("/instances/missing/R1").status_code == 404
    # neural models are chosen by name from the checkpoints folder, never by path
    for model in (None, "../../etc/passwd", "C:/secret.pt"):
        req = {"split": "sample", "route_id": "R1", "solver": "attention", "model": model}
        assert client.post("/solve", json=req).status_code == 404


def test_neural_solver_through_the_api(amazon_raw: Path, tmp_path: Path) -> None:
    run("sample", raw_root=amazon_raw / "raw", out_root=amazon_raw / "processed")
    ckpt_dir = tmp_path / "ckpt"
    name = tiny_model(ckpt_dir)
    api = TestClient(create_app(data_root=amazon_raw / "processed", checkpoint_dir=ckpt_dir))
    models = api.get("/models").json()
    assert [m["name"] for m in models] == [name] and models[0]["num_customers"] == 6
    sol = api.post("/solve", json={"split": "sample", "route_id": "R2", "solver": "attention_ls",
                                   "model": name}).json()  # fmt: skip
    assert sol["model"] == name and sorted(v for r in sol["routes"] for v in r) == [1, 2, 3]
