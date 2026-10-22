"""Make the demo's link preview (web/public/og.png) and the README GIF (docs/media/demo.gif)
from the exported demo data, so both always show the same numbers as the page.

    uv run python scripts/make_media.py [--route RouteID_...]

By default the route is the demo route whose saving against the driver is closest to the average
over all test routes: a typical route, not the best one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from PIL import Image

DATA = Path("web/public/data")
BG, TEXT, MUTED, UNVISITED = "#faf9f5", "#1f1e1d", "#5f5e5a", "#c9c7bd"
COLORS = {"driver": "#BA7517", "ours": "#534AB7"}
MINUS = chr(0x2212)  # typographic minus, as on the page
NAMES = {"driver": "Real driver", "ours": "Our model"}


def hours(h: float) -> str:
    m = round(h * 60)
    return f"{m // 60} h {m % 60:02d} min"


def typical_route(index: dict[str, Any]) -> str:
    s = index["headline"]["solvers"]
    target = 1 - s["ours"]["travel_h"] / s["driver"]["travel_h"]
    saving = {r["id"]: 1 - r["hours"]["ours"] / r["hours"]["driver"] for r in index["routes"]}
    return min(saving, key=lambda rid: abs(saving[rid] - target))


class Track:
    """One van's path in plot coordinates (lng scaled by cos(lat), lat) with arrival times."""

    def __init__(self, doc: dict[str, Any], key: str) -> None:
        sol = doc[key] if key in doc else doc["solvers"][key]
        self.kx = math.cos(math.radians(doc["depot"][0]))
        home = self.xy(doc["depot"])
        route, times = sol["routes"][0], sol["times"][0]
        self.points = np.array([home, *(self.xy(doc["stops"][n - 1]) for n in route), home])
        self.times = np.array([0, *times], dtype=float)
        self.route = route
        self.travel_h = sol["travel_h"]

    def xy(self, latlng: list[float]) -> tuple[float, float]:
        return latlng[1] * self.kx, latlng[0]

    def driven(self, t: float) -> np.ndarray:
        i = int(np.searchsorted(self.times, t, side="right"))
        if i >= len(self.times):
            return self.points
        f = (t - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        here = self.points[i - 1] + f * (self.points[i] - self.points[i - 1])
        return np.vstack([self.points[:i], here])


def draw_route(ax: Axes, doc: dict[str, Any], track: Track, color: str, t: float) -> None:
    stops = np.array([track.xy(s) for s in doc["stops"]])
    arrived = np.zeros(len(stops), dtype=bool)
    for node, at in zip(track.route, track.times[1:-1], strict=True):
        arrived[node - 1] = at <= t
    ax.scatter(*stops[~arrived].T, s=14, c=UNVISITED, edgecolors="white", linewidths=0.5, zorder=2)
    path = track.driven(t)
    ax.plot(*path.T, color=color, lw=1.8, alpha=0.9, zorder=3, solid_capstyle="round")
    ax.scatter(*stops[arrived].T, s=16, c=color, edgecolors="white", linewidths=0.5, zorder=4)
    if t < track.times[-1]:
        ax.scatter(*path[-1], s=70, c=color, edgecolors="white", linewidths=1.5, zorder=5)
    pad = 0.08 * np.ptp(stops, axis=0)
    ax.set_xlim(stops[:, 0].min() - pad[0], stops[:, 0].max() + pad[0])
    ax.set_ylim(stops[:, 1].min() - pad[1], stops[:, 1].max() + pad[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#e3e1d8")


def to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
    return Image.fromarray(rgba[..., :3].copy())


def make_gif(doc: dict[str, Any], out: Path, frames: int = 72, hold: int = 18) -> None:
    tracks = {k: Track(doc, k) for k in COLORS}
    end = max(t.times[-1] for t in tracks.values())
    images = []
    for f in range(frames + hold):
        t = end * min(1.0, f / (frames - 1))
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.9), dpi=100, facecolor=BG)
        for ax, (key, track) in zip(axes, tracks.items(), strict=True):
            ax.set_facecolor(BG)
            draw_route(ax, doc, track, COLORS[key], t)
            done = t >= track.times[-1]
            ax.set_title(f"{NAMES[key]}  ·  {hours(track.travel_h)} of driving"
                         + ("  ✓" if done else ""), color=COLORS[key], fontsize=12,
                         fontweight="bold", loc="left")  # fmt: skip
        clock = f"{int(t // 3600)}:{int(t % 3600 // 60):02d}"
        saving = 100 * (1 - tracks["ours"].travel_h / tracks["driver"].travel_h)
        where = f"{doc['city']}, {len(doc['stops'])} stops"
        fig.text(0.02, 0.035, f"{where}  ·  {clock} into the shift", color=MUTED, fontsize=10)
        fig.text(0.98, 0.035, f"same stops, {saving:.1f} % less driving", color=TEXT,
                 fontsize=10, ha="right")  # fmt: skip
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.05)
        images.append(to_image(fig).quantize(colors=48, method=Image.Quantize.MEDIANCUT))
        plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=70, loop=0,
                   optimize=True)  # fmt: skip


def make_preview(index: dict[str, Any], doc: dict[str, Any], out: Path) -> None:
    s = index["headline"]["solvers"]
    days = index["days"]
    day_saving = 100 * (1 - sum(d["hours"]["ours"] for d in days)
                        / sum(d["hours"]["driver"] for d in days))  # fmt: skip
    stats = [
        (f"{MINUS}{100 * (1 - s['ours']['travel_h'] / s['driver']['travel_h']):.1f} %",
         "driving time vs real drivers"),
        (f"{s['pyvrp_1s']['ours_wins']} / {index['headline']['routes']}",
         "routes better than PyVRP at equal time"),
        (f"{MINUS}{day_saving:.1f} %", "fleet hours on whole station-days"),
    ]  # fmt: skip
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
    fig.text(0.05, 0.84, "Neural last-mile routing", fontsize=38, fontweight="bold", color=TEXT)
    fig.text(0.05, 0.74, "A Transformer trained with reinforcement learning\n"
             "plans real Amazon delivery routes, against real\ndrivers, OR-Tools and PyVRP.",
             fontsize=16, color=MUTED, va="top", linespacing=1.4)  # fmt: skip
    for i, (big, label) in enumerate(stats):
        y = 0.40 - i * 0.12
        fig.text(0.05, y, big, fontsize=30, fontweight="bold", color=COLORS["ours"])
        fig.text(0.255, y + 0.012, label, fontsize=16, color=TEXT)
    fig.text(0.05, 0.05, "marcofer14.github.io/GraphProject", fontsize=14, color=MUTED)
    ax = fig.add_axes((0.66, 0.1, 0.31, 0.78))
    ax.set_facecolor(BG)
    track = Track(doc, "ours")
    draw_route(ax, doc, track, COLORS["ours"], math.inf)
    ax.set_title(f"Our model on a real {doc['city']} route", color=MUTED, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_image(fig).save(out, optimize=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", default=None, help="demo route id (default: a typical one)")
    args = parser.parse_args()
    index = json.loads((DATA / "index.json").read_text())
    rid = args.route or typical_route(index)
    doc = json.loads((DATA / "routes" / f"{rid}.json").read_text())
    make_preview(index, doc, Path("web/public/og.png"))
    make_gif(doc, Path("docs/media/demo.gif"))
    for p in (Path("web/public/og.png"), Path("docs/media/demo.gif")):
        print(f"{p}  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
