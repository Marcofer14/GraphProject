"""Plot validation curves of training runs.

    # one panel per experiment, x = wall-clock minutes (default)
    uv run python scripts/plot_training.py

    # ablation: several experiments on one axis, x = trained trajectories (fair across GPUs)
    uv run python scripts/plot_training.py --compare tsp20_pomo tsp20_rollout --x samples \\
        --out docs/figures/ablation_tsp20.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

XLABEL = {
    "minutes": "training minutes",
    "epoch": "epoch",
    "samples": "trained trajectories (millions)",
}


def latest_run(exp_dir: Path) -> Path | None:
    """Newest history.json under runs/<exp>/ (timestamped subfolders or a flat Colab run dir)."""
    found = sorted(exp_dir.glob("**/history.json"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def trajectories_per_step(run_dir: Path) -> float:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    train = cfg.get("train", {})
    batch = train.get("batch_size", 64)
    if train.get("baseline", "pomo") == "pomo":
        return float(batch * (train.get("num_starts") or cfg.get("num_customers", 20)))
    return float(batch)


def series(history_path: Path, x: str) -> tuple[list[float], list[float]]:
    hist = json.loads(history_path.read_text())
    ys = [h["val_length"] for h in hist]
    if x == "minutes":
        xs = [h["elapsed_s"] / 60 for h in hist]
    elif x == "epoch":
        xs = [h["epoch"] + 1 for h in hist]
    else:
        per_step = trajectories_per_step(history_path.parent)
        xs = [h["step"] * per_step / 1e6 for h in hist]
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--compare", nargs="+", help="experiments to overlay on one axis")
    parser.add_argument("--x", choices=list(XLABEL), default="minutes")
    parser.add_argument("--out", type=Path, default=Path("docs/figures/training_curves.png"))
    args = parser.parse_args()

    names = args.compare or sorted(p.name for p in args.runs.iterdir() if p.is_dir())
    runs = {n: h for n in names if (h := latest_run(args.runs / n))}
    missing = set(names) - set(runs)
    if missing:
        print(f"no history for: {', '.join(sorted(missing))}")
    if not runs:
        raise SystemExit("nothing to plot")

    if args.compare:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for name, path in runs.items():
            xs, ys = series(path, args.x)
            ax.plot(xs, ys, marker=".", label=f"{name} (best {min(ys):.4f})")
        ax.set(xlabel=XLABEL[args.x], ylabel="val tour length", title="Baseline ablation")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.6), squeeze=False)
        for ax, (name, path) in zip(axes[0], runs.items(), strict=True):
            xs, ys = series(path, args.x)
            ax.plot(xs, ys, marker=".")
            ax.set(title=name, xlabel=XLABEL[args.x], ylabel="val tour length")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
