from pathlib import Path

import numpy as np

from neural_routing.core import evaluate
from neural_routing.data.amazon import AmazonDataset, TravelTimeModel, haversine_km
from neural_routing.data.amazon_etl import run
from neural_routing.solvers import get_solver


def test_etl_and_route_instance(amazon_raw: Path) -> None:
    out = run("sample", raw_root=amazon_raw / "raw", out_root=amazon_raw / "processed")
    ds = AmazonDataset("sample", root=out.parent)
    assert len(ds) == 2

    inst = ds.route_instance("R1")
    assert inst.node_ids[0] == "ST" and inst.num_customers == 3
    np.testing.assert_allclose(inst.demands[1:], 1000.0)
    np.testing.assert_allclose(inst.service_times[1:], 60.0)
    # A1 has a 15:30-17:00 window, departure at 15:00 -> [1800, 7200]
    a1 = inst.node_ids.index("A1")
    np.testing.assert_allclose(inst.time_windows[a1], [1800.0, 7200.0])
    assert np.isinf(inst.time_windows[inst.node_ids.index("A0"), 1])
    # driver visited the stops in reverse id order
    assert inst.reference_routes == ((3, 2, 1),)
    m = evaluate(inst, get_solver("reference").solve(inst))
    # tour 0->3->2->1->0 with t(i, j) = 100 + 10|i - j|
    assert m.is_complete and m.travel_time_s == 130 + 110 + 110 + 110


def test_station_day_pooling(amazon_raw: Path) -> None:
    out = run("sample", raw_root=amazon_raw / "raw", out_root=amazon_raw / "processed")
    ds = AmazonDataset("sample", root=out.parent)
    model = ds.fit_travel_model(num_routes=2)
    inst = ds.station_day_instance("DXX1", "2018-07-20", model)
    assert inst.num_customers == 6 and inst.num_vehicles == 2
    # real within-route times are preserved
    np.testing.assert_allclose(inst.durations[:4, :4], ds.route_instance("R1").durations)
    assert inst.reference_routes is not None and len(inst.reference_routes) == 2
    m = evaluate(inst, get_solver("pyvrp").solve(inst, 0.2))
    assert m.is_complete


def test_travel_model_fit() -> None:
    km = np.array([0.0, 1.0, 2.0, 3.0])
    model = TravelTimeModel.fit(km, 30 + 120 * km)
    assert abs(model.slope - 120) < 1e-6 and abs(model.intercept - 30) < 1e-6
    assert model.predict(np.zeros((2, 2))).sum() == 0


def test_haversine_known_distance() -> None:
    # Madrid -> Barcelona is about 505 km
    d = haversine_km(np.array([[40.4168, -3.7038]]), np.array([[41.3874, 2.1686]]))
    assert 495 < d[0, 0] < 515
