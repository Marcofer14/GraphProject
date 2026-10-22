"""Training batches made of real route slices with their real travel-time sub-matrices.

A slice is a spatially coherent piece of one van's day: start at a random delivery zone of a
random route and add the nearest zones (by centroid) until ``N`` stops are collected. Everything
per route is precomputed once into numpy arrays, so a 32-slice batch costs a few milliseconds.

Units match the rest of the ML stack: coordinates are normalised (``normalise``), and travel
times, service times and windows are divided by ``travel_s_per_km x box size in km``, the same
seconds-per-unit the calibrated generator uses.

Routes are split deterministically by id into training and validation subsets, so validation
slices never come from routes seen in training; the *evaluation* split stays untouched for the
final benchmark.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from neural_routing.data.amazon import AmazonDataset
from neural_routing.data.calibration import project_km
from neural_routing.ml.env import Batch
from neural_routing.ml.generators import load_params, normalise


@dataclass
class _Route:
    xy: np.ndarray  # (n, 2) km from the station, station first
    node: np.ndarray  # (n,) matrix index of each row
    zone: np.ndarray  # (n,) zone code per stop (station -1)
    centres: np.ndarray  # (z, 2) zone centroids
    demand: np.ndarray  # (n,) fraction of the van's capacity
    service: np.ndarray  # (n,) seconds
    tw: np.ndarray  # (n, 2) seconds after departure, [0, inf] without window
    route_id: str


def in_validation(route_id: str, fraction: float) -> bool:
    """Stable, uniform assignment of a route to the validation subset."""
    h = int(hashlib.sha1(route_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < fraction


class RealRouteSampler:
    def __init__(
        self,
        ds: AmazonDataset,
        subset: str = "train",
        val_fraction: float = 0.1,
        min_stops: int = 20,
    ) -> None:
        if subset not in ("train", "val", "all"):
            raise ValueError(f"unknown subset {subset!r}")
        self.ds = ds
        self.speed = load_params().travel_s_per_km
        km = project_km(ds.stops)[["route_id", "stop_id", "x", "y"]]
        stops = ds.stops.merge(km, on=["route_id", "stop_id"], how="left")  # station: NaN -> 0
        stops = stops.sort_values(["route_id", "node"])
        capacity = ds.routes["capacity_cm3"]
        self.routes: list[_Route] = []
        for rid, g in stops.groupby("route_id", sort=False):
            rid = str(rid)
            if subset != "all" and in_validation(rid, val_fraction) != (subset == "val"):
                continue
            if len(g) - 1 < min_stops:
                continue
            xy = g[["x", "y"]].fillna(0.0).to_numpy()
            zone_names = g["zone_id"].fillna("?").to_numpy()
            codes = {z: i for i, z in enumerate(dict.fromkeys(zone_names[1:]))}
            zone = np.array([-1] + [codes[z] for z in zone_names[1:]])
            centres = np.stack([xy[1:][zone[1:] == i].mean(0) for i in range(len(codes))])
            tw = np.column_stack(
                [g["tw_start_s"].fillna(0).clip(lower=0), g["tw_end_s"].fillna(np.inf)]
            ).astype(np.float64)
            tw[0] = (0.0, np.inf)
            self.routes.append(
                _Route(
                    xy=xy,
                    node=g["node"].to_numpy(),
                    zone=zone,
                    centres=centres,
                    demand=(g["volume_cm3"] / capacity[rid]).fillna(0).to_numpy(),
                    service=g["service_s"].fillna(0).to_numpy(),
                    tw=tw,
                    route_id=rid,
                )
            )
        if not self.routes:
            raise ValueError("no routes left for this subset and size")

    def __len__(self) -> int:
        return len(self.routes)

    def _slice(self, r: _Route, n: int, rng: np.random.Generator) -> np.ndarray:
        """Row indices (station first) of a coherent n-stop slice of route r."""
        seed = rng.integers(len(r.centres))
        order = np.argsort(((r.centres - r.centres[seed]) ** 2).sum(1))
        picked: list[np.ndarray] = []
        taken = 0
        for z in order:
            rows = np.flatnonzero(r.zone == z)
            need = n - taken
            if len(rows) > need:
                rows = rng.choice(rows, need, replace=False)
            picked.append(rows)
            taken += len(rows)
            if taken >= n:
                break
        return np.concatenate([[0], *picked])

    def sample(
        self,
        batch_size: int,
        num_customers: int,
        problem: str = "cvrp",
        normalisation: str = "customers",
        rng: np.random.Generator | None = None,
        with_time: bool = False,
    ) -> Batch:
        rng = rng or np.random.default_rng()
        eligible = [r for r in self.routes if len(r.xy) - 1 >= num_customers]
        if not eligible:
            raise ValueError(f"no route has {num_customers} stops")
        xy, dist, demand, service, tw = [], [], [], [], []
        for k in rng.integers(len(eligible), size=batch_size):
            r = eligible[int(k)]
            rows = self._slice(r, num_customers, rng)
            xy.append(r.xy[rows])
            D = self.ds.durations(r.route_id)
            dist.append(D[np.ix_(r.node[rows], r.node[rows])])
            demand.append(r.demand[rows])
            service.append(r.service[rows])
            tw.append(r.tw[rows])
        locs, span_km = normalise(torch.tensor(np.stack(xy), dtype=torch.float32), normalisation)
        unit = (self.speed * span_km)[:, None, None]  # seconds per normalised unit
        dem = torch.tensor(np.stack(demand), dtype=torch.float32).clamp(max=1.0)
        if problem == "tsp":
            dem = torch.zeros_like(dem)
        times: dict[str, torch.Tensor | None] = {"service": None, "tw": None}
        if with_time:
            times["service"] = torch.tensor(np.stack(service), dtype=torch.float32) / unit[..., 0]
            times["tw"] = torch.tensor(np.stack(tw), dtype=torch.float32) / unit
        return Batch(
            locs, dem, problem == "tsp", times["service"], times["tw"],
            torch.tensor(np.stack(dist), dtype=torch.float32) / unit,
        )  # fmt: skip
