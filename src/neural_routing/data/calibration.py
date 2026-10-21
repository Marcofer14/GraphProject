"""Fit the parameters of the Amazon-like instance generator from the real routes.

The unit of calibration is one *van route* (~150 stops in ~20 delivery zones), the size the
neural solver must eventually handle. Statistics are measured in kilometres on a local
equirectangular projection centred on each route's station:

* geometry - station-to-route-centroid distance, spread of zone centres around the centroid,
  spread of stops within a zone, stops per zone (all log-normal across routes);
* demand   - stop volume as a fraction of the van's capacity (log-normal);
* time     - service time per stop (log-normal); share of stops with a window, and the window
  start and width (empirical quantiles); travel speed from :class:`TravelTimeModel`.

Correlations between these quantities are ignored (they are sampled independently).

    uv run python -m neural_routing.data.calibration     # writes configs/generator/amazon.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neural_routing.data.amazon import AmazonDataset

DEFAULT_PATH = Path("configs/generator/amazon.json")
QUANTILES = np.linspace(0, 1, 21)


@dataclass(frozen=True)
class LogNormal:
    mu: float
    sigma: float

    @classmethod
    def fit(cls, values: np.ndarray) -> LogNormal:
        logs = np.log(values[values > 0])
        return cls(float(logs.mean()), float(logs.std()))

    @property
    def median(self) -> float:
        return float(np.exp(self.mu))


@dataclass(frozen=True)
class GeneratorParams:
    depot_distance_km: LogNormal
    zone_spread_km: LogNormal
    stop_spread_km: LogNormal
    stops_per_zone: LogNormal
    demand_fraction: LogNormal
    service_s: LogNormal
    tw_probability: float
    tw_start_s_quantiles: list[float]
    tw_width_s_quantiles: list[float]
    travel_s_per_km: float
    travel_intercept_s: float
    routes_used: int
    # zone spread was fitted on whole routes of this many stops; smaller instances cover fewer,
    # neighbouring zones, so the generator scales the spread by sqrt(N / route_stops)
    route_stops: float = 150.0

    def save(self, path: Path = DEFAULT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> GeneratorParams:
        raw = json.loads(Path(path).read_text())
        for key in ("depot_distance_km", "zone_spread_km", "stop_spread_km", "stops_per_zone",
                    "demand_fraction", "service_s"):  # fmt: skip
            raw[key] = LogNormal(**raw[key])
        return cls(**raw)


def project_km(stops: pd.DataFrame) -> pd.DataFrame:
    """Add x/y in km relative to each route's station; returns drop-offs only."""
    station = stops[stops["type"] == "Station"].set_index("route_id")[["lat", "lng"]]
    d = stops[stops["type"] == "Dropoff"].join(station, on="route_id", rsuffix="_st")
    d = d.assign(
        x=(d["lng"] - d["lng_st"]) * 111.32 * np.cos(np.radians(d["lat_st"])),
        y=(d["lat"] - d["lat_st"]) * 110.574,
        zone=d["zone_id"].fillna("?"),
    )
    return d


def route_geometry(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in d.groupby("route_id", sort=False):
        xy = g[["x", "y"]].to_numpy()
        centroid = xy.mean(0)
        centres = g.groupby("zone")[["x", "y"]].mean()
        within = xy - centres.loc[g["zone"]].to_numpy()
        rows.append(
            {
                "route_id": rid,
                "stops": len(g),
                "depot_distance_km": float(np.linalg.norm(centroid)),
                "zone_spread_km": float(
                    np.sqrt(((centres.to_numpy() - centroid) ** 2).sum(1).mean())
                ),
                "stop_spread_km": float(np.sqrt((within**2).sum(1).mean())),
                "stops_per_zone": len(g) / len(centres),
            }
        )
    return pd.DataFrame(rows)


def fit(ds: AmazonDataset) -> GeneratorParams:
    d = project_km(ds.stops)
    geo = route_geometry(d)
    capacity = d["route_id"].map(ds.routes["capacity_cm3"])
    tw = d.dropna(subset=["tw_end_s"])
    model = ds.fit_travel_model()
    return GeneratorParams(
        depot_distance_km=LogNormal.fit(geo["depot_distance_km"].to_numpy()),
        zone_spread_km=LogNormal.fit(geo["zone_spread_km"].to_numpy()),
        stop_spread_km=LogNormal.fit(geo["stop_spread_km"].to_numpy()),
        stops_per_zone=LogNormal.fit(geo["stops_per_zone"].to_numpy()),
        demand_fraction=LogNormal.fit((d["volume_cm3"] / capacity).to_numpy()),
        service_s=LogNormal.fit(d["service_s"].to_numpy()),
        tw_probability=float(len(tw) / len(d)),
        tw_start_s_quantiles=np.quantile(tw["tw_start_s"], QUANTILES).round(1).tolist(),
        tw_width_s_quantiles=np.quantile(tw["tw_end_s"] - tw["tw_start_s"], QUANTILES)
        .round(1)
        .tolist(),
        travel_s_per_km=model.slope,
        travel_intercept_s=model.intercept,
        routes_used=len(geo),
        route_stops=float(geo["stops"].median()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit configs/generator/amazon.json")
    parser.add_argument("--split", default="training")
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    params = fit(AmazonDataset(args.split))
    params.save(args.out)
    print(json.dumps(asdict(params), indent=2))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
