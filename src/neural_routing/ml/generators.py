"""Amazon-like instance generator, calibrated on real routes (``configs/generator/amazon.json``).

One instance looks like one van's day: a station far from the delivery area, customers grouped
in delivery zones, tiny demands (capacity rarely binds unless ``stops_per_vehicle`` is set) and
mostly loose time windows. Coordinates are rescaled to the unit square exactly like real instances
are at inference (:func:`neural_routing.solvers.neural.to_batch`), so train and test geometry match.

Time features are returned in *normalised time units*: one unit is the time needed to drive one
unit of normalised distance (``seconds_per_unit`` per instance), so they can be compared directly
with Euclidean distances in the environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from neural_routing.data.calibration import DEFAULT_PATH, GeneratorParams, LogNormal
from neural_routing.ml.env import Batch


@dataclass
class TimeFeatures:
    service: Tensor  # (B, N+1) normalised time units; depot 0
    tw: Tensor  # (B, N+1, 2) [earliest, latest]; no window = [0, inf]
    seconds_per_unit: Tensor  # (B,)

    def to(self, device: torch.device | str) -> TimeFeatures:
        return TimeFeatures(
            self.service.to(device), self.tw.to(device), self.seconds_per_unit.to(device)
        )


@cache
def load_params(path: str | Path = DEFAULT_PATH) -> GeneratorParams:
    return GeneratorParams.load(Path(path))


def _lognormal(p: LogNormal, shape: tuple[int, ...], g: torch.Generator | None) -> Tensor:
    return torch.exp(p.mu + p.sigma * torch.randn(shape, generator=g))


def _from_quantiles(q: list[float], shape: tuple[int, ...], g: torch.Generator | None) -> Tensor:
    """Inverse-CDF sampling with linear interpolation between stored quantiles."""
    table = torch.tensor(q, dtype=torch.float32)
    pos = torch.rand(shape, generator=g) * (len(q) - 1)
    lo = pos.floor().long().clamp(max=len(q) - 2)
    return table[lo] + (table[lo + 1] - table[lo]) * (pos - lo)


NORMALISATIONS = ("all", "customers")


def normalise(locs_km: Tensor, mode: str = "all") -> tuple[Tensor, Tensor]:
    """Shift and scale (aspect ratio kept) so the reference nodes span [0, 1]. Returns (locs, km).

    ``all`` uses every node, so a far depot squashes the customers into a corner. ``customers``
    uses the customers only: they fill the square and the depot may lie outside it (distances stay
    exact, only the unit changes); pair it with ``ModelConfig.depot_encoding="polar"``.
    """
    if mode not in NORMALISATIONS:
        raise ValueError(f"unknown normalisation {mode!r}")
    ref = locs_km if mode == "all" else locs_km[:, 1:]
    mins = ref.min(dim=1, keepdim=True).values
    span = (ref.max(dim=1, keepdim=True).values - mins).amax(dim=-1).clamp(min=1e-9)
    return (locs_km - mins) / span[..., None], span.squeeze(-1)


def generate_amazon_like(
    batch_size: int,
    num_customers: int,
    problem: str = "cvrp",
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    params: GeneratorParams | None = None,
    stops_per_vehicle: float | None = None,
    normalisation: str = "all",
) -> tuple[Batch, TimeFeatures]:
    """``stops_per_vehicle`` rescales demands so a van fills up after that many average stops
    (None keeps the real, almost never binding, demands). See :func:`normalise`."""
    p = params or load_params()
    B, N, g = batch_size, num_customers, generator

    # geometry in km, customers' zone centres around the origin, station far away
    depot_r = _lognormal(p.depot_distance_km, (B,), g)
    angle = torch.rand(B, generator=g) * 2 * math.pi
    depot = torch.stack([depot_r * torch.cos(angle), depot_r * torch.sin(angle)], dim=-1)
    # spreads were fitted as 2-D RMS distances; a per-axis sd is that / sqrt(2). Area grows with
    # the number of zones covered, so the zone spread also scales with sqrt(stops)
    zone_sd = _lognormal(p.zone_spread_km, (B,), g) * math.sqrt(N / p.route_stops / 2)
    stop_sd = _lognormal(p.stop_spread_km, (B,), g) / math.sqrt(2)
    zones = (N / _lognormal(p.stops_per_zone, (B,), g)).round().clamp(1, N)
    centres = torch.randn(B, N, 2, generator=g) * zone_sd[:, None, None]
    zone_of = (torch.rand(B, N, generator=g) * zones[:, None]).long()
    customers = centres.gather(1, zone_of[..., None].expand(B, N, 2))
    customers = customers + torch.randn(B, N, 2, generator=g) * stop_sd[:, None, None]
    locs, span_km = normalise(torch.cat([depot[:, None], customers], dim=1), normalisation)

    # demand as a fraction of capacity
    demand = torch.zeros(B, N + 1)
    if problem == "cvrp":
        frac = _lognormal(p.demand_fraction, (B, N), g)
        if stops_per_vehicle:
            mean = math.exp(p.demand_fraction.mu + p.demand_fraction.sigma**2 / 2)
            frac = frac / (mean * stops_per_vehicle)
        demand[:, 1:] = frac.clamp(max=1.0)
    elif problem != "tsp":
        raise ValueError(f"unknown problem {problem!r}")

    # time: service and windows in seconds, then in normalised units
    seconds_per_unit = p.travel_s_per_km * span_km
    service = torch.zeros(B, N + 1)
    service[:, 1:] = _lognormal(p.service_s, (B, N), g)
    tw = torch.zeros(B, N + 1, 2)
    tw[..., 1] = math.inf
    has_tw = torch.rand(B, N, generator=g) < p.tw_probability
    start = _from_quantiles(p.tw_start_s_quantiles, (B, N), g).clamp(min=0)
    end = start + _from_quantiles(p.tw_width_s_quantiles, (B, N), g)
    tw[:, 1:, 0] = torch.where(has_tw, start, torch.zeros_like(start))
    tw[:, 1:, 1] = torch.where(has_tw, end, torch.full_like(end, math.inf))
    times = TimeFeatures(
        service / seconds_per_unit[:, None], tw / seconds_per_unit[:, None, None], seconds_per_unit
    )
    return Batch(locs, demand, problem == "tsp").to(device), times.to(device)


def sample_real(
    stops_km: pd.DataFrame,
    capacity: pd.Series,
    count: int,
    num_customers: int,
    problem: str = "cvrp",
    seed: int = 0,
    travel_s_per_km: float | None = None,
    normalisation: str = "all",
) -> tuple[Batch, TimeFeatures]:
    """``count`` real sub-routes of ``num_customers`` stops, normalised like generated ones.

    A sub-route starts at a random zone of a random route and adds the nearest zones (by centroid)
    until it has enough stops, i.e. a spatially coherent slice of one van's day.
    ``stops_km`` is the output of :func:`neural_routing.data.calibration.project_km`.
    """
    rng = np.random.default_rng(seed)
    by_route: dict[str, pd.DataFrame] = {
        str(rid): g for rid, g in stops_km.groupby("route_id", sort=False)
    }
    eligible = [rid for rid, g in by_route.items() if len(g) >= num_customers]
    speed = travel_s_per_km or load_params().travel_s_per_km
    locs, demand, service, tw = [], [], [], []
    for pick in rng.integers(len(eligible), size=count):
        rid = eligible[int(pick)]
        g = by_route[rid]
        centres = g.groupby("zone")[["x", "y"]].mean()
        seed_zone = centres.index[int(rng.integers(len(centres)))]
        order = ((centres - centres.loc[seed_zone]) ** 2).sum(axis=1).sort_values().index
        picked: list[pd.DataFrame] = []
        taken = 0
        for z in order:
            zone_stops = g[g["zone"] == z]
            need = num_customers - taken
            if len(zone_stops) > need:
                zone_stops = zone_stops.sample(need, random_state=int(rng.integers(2**31)))
            picked.append(zone_stops)
            taken += len(zone_stops)
            if taken >= num_customers:
                break
        sub = pd.concat(picked)
        xy = np.vstack([[0.0, 0.0], sub[["x", "y"]].to_numpy()])  # station at the origin
        locs.append(xy)
        frac = (
            (sub["volume_cm3"] / capacity[rid]).to_numpy() if problem == "cvrp" else 0 * xy[1:, 0]
        )
        demand.append(np.r_[0.0, frac])
        service.append(np.r_[0.0, sub["service_s"].to_numpy()])
        start = sub["tw_start_s"].fillna(0).clip(lower=0).to_numpy()
        end = sub["tw_end_s"].fillna(np.inf).to_numpy()
        tw.append(np.vstack([[0.0, np.inf], np.column_stack([start, end])]))
    xy_t, span_km = normalise(torch.tensor(np.stack(locs), dtype=torch.float32), normalisation)
    seconds_per_unit = speed * span_km
    times = TimeFeatures(
        torch.tensor(np.stack(service), dtype=torch.float32) / seconds_per_unit[:, None],
        torch.tensor(np.stack(tw), dtype=torch.float32) / seconds_per_unit[:, None, None],
        seconds_per_unit,
    )
    dem = torch.tensor(np.stack(demand), dtype=torch.float32).clamp(max=1.0)
    return Batch(xy_t, dem, problem == "tsp"), times
