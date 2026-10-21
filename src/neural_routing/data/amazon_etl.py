"""ETL: Amazon Last Mile Routing Challenge JSON -> tidy Parquet tables + one memory-mappable matrix file.

Output (``data/processed/<split>/``):

* ``routes.parquet``   one row per route (station, date, departure, capacity, driver score, matrix offset)
* ``stops.parquet``    one row per stop, ``node`` = matrix index (0 = station), stop-level demand,
                       service time and time window aggregated from its packages, driver's visit order
* ``packages.parquet`` one row per package (dimensions, service time, time window)
* ``durations.f32``    all travel-time matrices, float32, row-major, concatenated; route r occupies
                       ``[matrix_offset, matrix_offset + n_nodes**2)``

Times are seconds relative to the route departure. The 1.7 GB travel-time file is streamed with
ijson so peak memory stays low; the other files contain NaN literals, which only ``json`` accepts.

Usage::

    uv run nr-etl --split training
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import ijson
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

RAW_ROOT = Path("data/raw/almrrc2021")
OUT_ROOT = Path("data/processed")

SPLITS: dict[str, dict[str, str]] = {
    "training": {
        "route": "almrrc2021-data-training/model_build_inputs/route_data.json",
        "package": "almrrc2021-data-training/model_build_inputs/package_data.json",
        "travel": "almrrc2021-data-training/model_build_inputs/travel_times.json",
        "actual": "almrrc2021-data-training/model_build_inputs/actual_sequences.json",
    },
    "evaluation": {
        "route": "almrrc2021-data-evaluation/model_apply_inputs/eval_route_data.json",
        "package": "almrrc2021-data-evaluation/model_apply_inputs/eval_package_data.json",
        "travel": "almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json",
        "actual": "almrrc2021-data-evaluation/model_score_inputs/eval_actual_sequences.json",
    },
    # the 13-route sample shipped with the training data; handy for quick runs and tests
    "sample": {
        "route": "almrrc2021-data-training/model_apply_inputs/new_route_data.json",
        "package": "almrrc2021-data-training/model_apply_inputs/new_package_data.json",
        "travel": "almrrc2021-data-training/model_apply_inputs/new_travel_times.json",
        "actual": "almrrc2021-data-training/model_score_inputs/new_actual_sequences.json",
    },
}


def _load_json(path: Path) -> Any:
    with path.open("rb") as f:
        return json.load(f)


def build_routes_and_stops(
    route_data: dict[str, Any], actual: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes, stops = [], []
    for route_id, r in route_data.items():
        stop_items = r["stops"]
        station = [s for s, v in stop_items.items() if v["type"] == "Station"]
        if len(station) != 1:
            log.warning("route %s has %d stations, skipped", route_id, len(station))
            continue
        order = station + sorted(s for s in stop_items if s != station[0])
        driver = actual.get(route_id, {}).get("actual", {})
        for node, stop_id in enumerate(order):
            v = stop_items[stop_id]
            zone = v.get("zone_id")
            stops.append(
                {
                    "route_id": route_id,
                    "node": node,
                    "stop_id": stop_id,
                    "lat": v["lat"],
                    "lng": v["lng"],
                    "type": v["type"],
                    "zone_id": zone if isinstance(zone, str) else None,
                    "actual_order": driver.get(stop_id, -1),
                }
            )
        routes.append(
            {
                "route_id": route_id,
                "station_code": r["station_code"],
                "date": r["date_YYYY_MM_DD"],
                "departure_utc": pd.Timestamp(f"{r['date_YYYY_MM_DD']} {r['departure_time_utc']}"),
                "capacity_cm3": float(r["executor_capacity_cm3"]),
                "route_score": r.get("route_score"),
                "n_nodes": len(order),
            }
        )
    return pd.DataFrame(routes), pd.DataFrame(stops)


def build_packages(package_data: dict[str, Any], routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route_id, stops in package_data.items():
        for stop_id, pkgs in stops.items():
            for pkg_id, p in pkgs.items():
                tw, dim = p["time_window"], p["dimensions"]
                rows.append(
                    (
                        route_id,
                        stop_id,
                        pkg_id,
                        p.get("scan_status"),
                        p["planned_service_time_seconds"],
                        tw["start_time_utc"],
                        tw["end_time_utc"],
                        dim["depth_cm"],
                        dim["height_cm"],
                        dim["width_cm"],
                    )
                )
    cols = ["route_id", "stop_id", "package_id", "scan_status", "service_s",
            "tw_start", "tw_end", "depth_cm", "height_cm", "width_cm"]  # fmt: skip
    df = pd.DataFrame(rows, columns=cols)
    df["volume_cm3"] = df["depth_cm"] * df["height_cm"] * df["width_cm"]
    dep = df["route_id"].map(routes.set_index("route_id")["departure_utc"])
    for col in ("tw_start", "tw_end"):
        ts = pd.to_datetime(df[col].where(df[col].map(lambda x: isinstance(x, str))), errors="coerce")
        df[col + "_s"] = (ts - dep).dt.total_seconds()
    return df.drop(columns=["tw_start", "tw_end"])


def aggregate_stop_windows(packages: pd.DataFrame) -> pd.DataFrame:
    """Stop window = intersection of its packages' windows; if empty, the earliest-ending one."""
    g = packages.groupby(["route_id", "stop_id"])
    agg = g.agg(
        n_packages=("package_id", "size"),
        volume_cm3=("volume_cm3", "sum"),
        service_s=("service_s", "sum"),
        tw_start_s=("tw_start_s", "max"),
        tw_end_s=("tw_end_s", "min"),
    )
    conflict = agg["tw_start_s"] > agg["tw_end_s"]
    if conflict.any():
        tw = packages.dropna(subset=["tw_end_s"]).sort_values("tw_end_s")
        first = tw.groupby(["route_id", "stop_id"]).first()
        idx = agg.index[conflict]
        agg.loc[idx, "tw_start_s"] = first.loc[idx, "tw_start_s"]
        agg.loc[idx, "tw_end_s"] = first.loc[idx, "tw_end_s"]
        log.info("%d stops had conflicting package windows", int(conflict.sum()))
    return agg.reset_index()


def write_matrices(travel_path: Path, stops: pd.DataFrame, out_path: Path) -> dict[str, int]:
    """Stream travel times route by route into one float32 file; returns route -> element offset."""
    order = stops.groupby("route_id")["stop_id"].apply(list).to_dict()
    offsets: dict[str, int] = {}
    offset = 0
    with travel_path.open("rb") as f, out_path.open("wb") as out:
        for route_id, tt in tqdm(ijson.kvitems(f, "", use_float=True), desc="travel times"):
            ids = order.get(route_id)
            if ids is None:
                continue
            M = np.array([[tt[a][b] for b in ids] for a in ids], dtype=np.float32)
            out.write(M.tobytes())
            offsets[route_id] = offset
            offset += M.size
    return offsets


def run(split: str, raw_root: Path = RAW_ROOT, out_root: Path = OUT_ROOT) -> Path:
    files = {k: raw_root / v for k, v in SPLITS[split].items()}
    out = out_root / split
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    log.info("routes + stops")
    routes, stops = build_routes_and_stops(_load_json(files["route"]), _load_json(files["actual"]))

    log.info("packages")
    packages = build_packages(_load_json(files["package"]), routes)
    stop_agg = aggregate_stop_windows(packages)
    stops = stops.merge(stop_agg, on=["route_id", "stop_id"], how="left")
    stops[["n_packages", "volume_cm3", "service_s"]] = (
        stops[["n_packages", "volume_cm3", "service_s"]].fillna(0)
    )

    log.info("travel-time matrices")
    offsets = write_matrices(files["travel"], stops, out / "durations.f32")
    routes["matrix_offset"] = routes["route_id"].map(offsets)
    missing = routes["matrix_offset"].isna()
    if missing.any():
        log.warning("%d routes without travel times dropped", int(missing.sum()))
        keep = set(routes.loc[~missing, "route_id"])
        routes, stops = routes[~missing], stops[stops["route_id"].isin(keep)]
        packages = packages[packages["route_id"].isin(keep)]
    routes["matrix_offset"] = routes["matrix_offset"].astype("int64")

    routes.to_parquet(out / "routes.parquet", index=False)
    stops.to_parquet(out / "stops.parquet", index=False)
    packages.to_parquet(out / "packages.parquet", index=False)
    log.info(
        "%s: %d routes, %d stops, %d packages in %.0fs",
        split, len(routes), len(stops), len(packages), time.perf_counter() - t0,
    )  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon Last Mile JSON -> Parquet")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="sample")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    splits = ["training", "evaluation"] if args.split == "all" else [args.split]
    for split in splits:
        run(split, args.raw_root, args.out_root)


if __name__ == "__main__":
    main()
