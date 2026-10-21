"""Turn the processed Amazon tables into :class:`VRPInstance` objects.

Two problem views of the same data:

* :meth:`AmazonDataset.route_instance` - one van, its stops, the real travel-time matrix and the
  driver's actual sequence as reference (single-vehicle TSP with time windows).
* :meth:`AmazonDataset.station_day_instance` - every stop served from one station on one day,
  re-assigned from scratch to the same number of vans (CVRPTW). Amazon only publishes travel
  times *within* each route, so cross-route pairs come from a haversine model fitted on the
  within-route data (see :class:`TravelTimeModel`). Real times are kept wherever they exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_routing.core.instance import FloatArray, VRPInstance

EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise great-circle distance between (n, 2) and (m, 2) lat/lng arrays -> (n, m)."""
    la1, lo1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    la2, lo2 = np.radians(b[:, 0])[None], np.radians(b[:, 1])[None]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return np.asarray(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h)), dtype=np.float64)


@dataclass(frozen=True)
class TravelTimeModel:
    """seconds ~= intercept + slope * haversine_km (least squares on observed pairs)."""

    intercept: float
    slope: float

    def predict(self, km: FloatArray) -> FloatArray:
        out = np.asarray(self.intercept + self.slope * km, dtype=np.float64)
        out[km == 0] = 0.0
        return out

    @classmethod
    def fit(cls, km: FloatArray, seconds: FloatArray) -> TravelTimeModel:
        mask = km > 0
        slope, intercept = np.polyfit(km[mask], seconds[mask], deg=1)
        return cls(intercept=float(intercept), slope=float(slope))


class AmazonDataset:
    def __init__(self, split: str = "training", root: Path = Path("data/processed")) -> None:
        self.dir = root / split
        if not (self.dir / "routes.parquet").exists():
            raise FileNotFoundError(f"{self.dir} not found, run `uv run nr-etl --split {split}`")
        self.routes = pd.read_parquet(self.dir / "routes.parquet").set_index("route_id")
        self.stops = pd.read_parquet(self.dir / "stops.parquet")
        self._by_route = {rid: df for rid, df in self.stops.groupby("route_id", sort=False)}

    def __len__(self) -> int:
        return len(self.routes)

    @cached_property
    def _matrices(self) -> np.memmap[Any, np.dtype[np.float32]]:
        return np.memmap(self.dir / "durations.f32", dtype=np.float32, mode="r")

    def _route_row(self, route_id: str) -> dict[str, Any]:
        return {str(k): v for k, v in self.routes.loc[route_id].items()}

    def durations(self, route_id: str) -> FloatArray:
        r = self._route_row(route_id)
        n, off = int(r["n_nodes"]), int(r["matrix_offset"])
        D = np.asarray(self._matrices[off : off + n * n], dtype=np.float64).reshape(n, n)
        np.fill_diagonal(D, 0.0)  # a few raw matrices have non-zero self travel times
        return D

    def route_instance(self, route_id: str) -> VRPInstance:
        r = self._route_row(route_id)
        s = self._by_route[route_id].sort_values("node")
        tw = np.column_stack(
            [s["tw_start_s"].fillna(0.0).clip(lower=0), s["tw_end_s"].fillna(np.inf)]
        ).astype(np.float64)
        tw[0] = (0.0, np.inf)
        order = s.loc[s["actual_order"] > 0].sort_values("actual_order")["node"]
        reference = (tuple(int(v) for v in order),) if len(order) == len(s) - 1 else None
        return VRPInstance(
            name=route_id,
            coords=s[["lat", "lng"]].to_numpy(np.float64),
            durations=self.durations(route_id),
            demands=s["volume_cm3"].to_numpy(np.float64),
            service_times=s["service_s"].to_numpy(np.float64),
            time_windows=tw,
            capacity=float(r["capacity_cm3"]),
            num_vehicles=1,
            node_ids=tuple(s["stop_id"]),
            reference_routes=reference,
            meta={
                "station": r["station_code"],
                "date": r["date"],
                "route_score": r["route_score"],
                "zones": tuple(s["zone_id"]),
            },
        )

    def fit_travel_model(self, num_routes: int = 300, seed: int = 0) -> TravelTimeModel:
        rng = np.random.default_rng(seed)
        size = min(num_routes, len(self))
        ids = rng.choice(self.routes.index.to_numpy(), size=size, replace=False)
        kms, secs = [], []
        for rid in ids:
            s = self._by_route[rid].sort_values("node")
            xy = s[["lat", "lng"]].to_numpy(np.float64)
            kms.append(haversine_km(xy, xy).ravel())
            secs.append(self.durations(rid).ravel())
        return TravelTimeModel.fit(np.concatenate(kms), np.concatenate(secs))

    def station_days(self) -> pd.DataFrame:
        g = self.routes.reset_index().groupby(["station_code", "date"])
        return g.agg(num_routes=("route_id", "size"), num_stops=("n_nodes", "sum")).reset_index()

    def station_day_instance(
        self, station: str, date: str, travel_model: TravelTimeModel
    ) -> VRPInstance:
        rids = self.routes.index[
            (self.routes["station_code"] == station) & (self.routes["date"] == date)
        ].tolist()
        if not rids:
            raise KeyError(f"no routes for {station} on {date}")
        parts = [self.route_instance(rid) for rid in rids]
        # node 0 = the (shared) station; each route's customers follow in turn
        coords = np.vstack([parts[0].coords[:1], *[p.coords[1:] for p in parts]])
        D = travel_model.predict(haversine_km(coords, coords))
        start = 1
        for p in parts:
            idx = np.r_[0, start : start + p.num_customers]
            D[np.ix_(idx, idx)] = p.durations  # real times where known
            start += p.num_customers

        def cat(attr: str) -> FloatArray:
            arrays = [getattr(p, attr)[1:] for p in parts]
            return np.concatenate([getattr(parts[0], attr)[:1], *arrays])

        node_ids = [parts[0].node_ids[0]]
        for rid, p in zip(rids, parts, strict=True):
            node_ids.extend(f"{rid}:{s}" for s in p.node_ids[1:])
        offsets = np.cumsum([0, *[p.num_customers for p in parts]])
        reference = tuple(
            tuple(int(v) + int(offsets[k]) for v in p.reference_routes[0])
            for k, p in enumerate(parts)
            if p.reference_routes
        )
        return VRPInstance(
            name=f"{station}_{date}",
            coords=coords,
            durations=D,
            demands=cat("demands"),
            service_times=cat("service_times"),
            time_windows=cat("time_windows"),
            capacity=float(np.median([p.capacity for p in parts])),
            num_vehicles=len(parts),
            node_ids=tuple(node_ids),
            reference_routes=reference if len(reference) == len(parts) else None,
            meta={"station": station, "date": date, "route_ids": tuple(rids)},
        )
