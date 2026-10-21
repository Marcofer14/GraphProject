from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_routing.data.amazon_etl import SPLITS


@pytest.fixture
def amazon_raw(tmp_path: Path) -> Path:
    """Two tiny routes on the same station/day, in the exact Amazon JSON layout (NaN included)."""
    nan = float("nan")
    stations = {"ST": {"lat": 47.60, "lng": -122.33, "type": "Station", "zone_id": nan}}

    def stops(prefix: str, dlat: float) -> dict[str, dict[str, object]]:
        return {
            f"{prefix}{k}": {"lat": 47.61 + dlat + 0.01 * k, "lng": -122.32, "type": "Dropoff",
                             "zone_id": f"Z-{prefix}"}
            for k in range(3)
        }  # fmt: skip

    route_data, package_data, travel, actual = {}, {}, {}, {}
    for rid, prefix, dlat in (("R1", "A", 0.0), ("R2", "B", 0.05)):
        s = stations | stops(prefix, dlat)
        route_data[rid] = {
            "station_code": "DXX1",
            "date_YYYY_MM_DD": "2018-07-20",
            "departure_time_utc": "15:00:00",
            "executor_capacity_cm3": 1_000_000.0,
            "route_score": "High",
            "stops": s,
        }
        ids = list(s)
        travel[rid] = {a: {b: 0.0 if a == b else 100.0 + 10 * abs(i - j)
                           for j, b in enumerate(ids)} for i, a in enumerate(ids)}  # fmt: skip
        package_data[rid] = {
            sid: {
                f"P-{sid}": {
                    "scan_status": "DELIVERED",
                    "time_window": (
                        {
                            "start_time_utc": "2018-07-20 15:30:00",
                            "end_time_utc": "2018-07-20 17:00:00",
                        }
                        if sid.endswith("1")
                        else {"start_time_utc": nan, "end_time_utc": nan}
                    ),
                    "planned_service_time_seconds": 60.0,
                    "dimensions": {"depth_cm": 10.0, "height_cm": 10.0, "width_cm": 10.0},
                }
            }
            for sid in ids
            if sid != "ST"
        }
        actual[rid] = {"actual": {sid: k for k, sid in enumerate(["ST", *reversed(ids[1:])])}}

    for name, payload in (("route", route_data), ("package", package_data),
                          ("travel", travel), ("actual", actual)):  # fmt: skip
        path = tmp_path / "raw" / SPLITS["sample"][name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))  # json writes NaN literals, like Amazon's files
    return tmp_path
