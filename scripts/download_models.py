"""Download the trained models from the GitHub release into checkpoints/.

uv run python scripts/download_models.py            # latest published tag below
uv run python scripts/download_models.py --tag v0.2.0
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

REPO = "Marcofer14/GraphProject"
MODELS = ["tsp20_pomo.pt", "cvrp20_pomo.pt", "cvrp50_pomo.pt"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v0.2.0")
    parser.add_argument("--dest", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        target = args.dest / name
        if target.exists():
            print(f"skip  {target} (exists)")
            continue
        url = f"https://github.com/{REPO}/releases/download/{args.tag}/{name}"
        print(f"get   {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, target.open("wb") as out:
            shutil.copyfileobj(resp, out)
    print("done")


if __name__ == "__main__":
    main()
