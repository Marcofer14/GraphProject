"""Plot validation curves of the latest run of each experiment -> docs/figures/training_curves.png

uv run python scripts/plot_training.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    runs = {}
    for exp in sorted(Path("runs").iterdir()):
        histories = sorted(exp.glob("*/history.json"))
        if histories:
            runs[exp.name] = json.loads(histories[-1].read_text())
    if not runs:
        raise SystemExit("no runs/*/*/history.json found")

    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.6), squeeze=False)
    for ax, (name, hist) in zip(axes[0], runs.items(), strict=True):
        ax.plot([h["elapsed_s"] / 60 for h in hist], [h["val_length"] for h in hist], marker=".")
        ax.set(title=name, xlabel="training minutes (GTX 1660 Super)", ylabel="val tour length")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path("docs/figures/training_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"written {out}")


if __name__ == "__main__":
    main()
