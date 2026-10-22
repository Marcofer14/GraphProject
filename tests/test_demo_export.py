import numpy as np
import pandas as pd

from neural_routing.core import VRPInstance
from neural_routing.eval.demo_export import city, pick_routes, schedule, solution_payload


def line_instance() -> VRPInstance:
    """Depot at 0, customers at 1, 2, 3 on a line (as in test_evaluate)."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    return VRPInstance(
        name="line",
        coords=np.column_stack([x, np.zeros(4)]),
        durations=np.abs(x[:, None] - x[None]),
        demands=np.array([0.0, 1.0, 1.0, 1.0]),
        service_times=np.array([0.0, 1.0, 1.0, 1.0]),
        time_windows=np.array([[0, np.inf], [5, np.inf], [0, np.inf], [0, 6]]),
        capacity=2.0,
    )


def test_schedule_matches_evaluator_clock() -> None:
    # arrive 1 -> wait to 5, serve to 6; reach 2 at 7, serve to 8; reach 3 at 9, back at 13
    assert schedule(line_instance(), [1, 2, 3]) == [5, 7, 9, 13]


def test_solution_payload() -> None:
    p = solution_payload(line_instance(), [[1, 2, 3]], runtime_s=0.1234)
    assert p["times"] == [[5, 7, 9, 13]]
    assert p["late"] == 1
    assert p["travel_h"] == round(6 / 3600, 3)
    assert p["runtime_s"] == 0.123


def test_pick_routes_per_city_is_seeded() -> None:
    df = pd.DataFrame(
        {
            "instance": [f"r{i}" for i in range(9)] * 2,  # two configs per route
            "station": ["DLA9"] * 6 + ["DSE5"] * 3 + ["DLA9"] * 6 + ["DSE5"] * 3,
        }
    )
    picked = pick_routes(df, per_city=2, seed=0)
    assert len(picked) == 4 and len(set(picked)) == 4
    assert sum(df.set_index("instance").loc[p, "station"].iloc[0] == "DSE5" for p in picked) == 2
    assert picked == pick_routes(df, per_city=2, seed=0)
    assert city("DLA9") == "Los Angeles" and city("XYZ1") == "XYZ1"


def test_moved_share() -> None:
    from neural_routing.eval.demo_export import moved_share

    drivers = [[1, 2, 3], [4, 5, 6]]
    assert moved_share(drivers, [[6, 5, 4], [3, 1, 2]]) == 0.0  # same groups, other order
    assert moved_share(drivers, [[1, 2, 4], [3, 5, 6]]) == round(2 / 6, 4)
