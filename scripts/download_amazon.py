"""Download the Amazon Last Mile Routing Research Challenge dataset (public S3 bucket).

Source: https://registry.opendata.aws/amazon-last-mile-challenges/  (license: CC BY-NC 4.0)

Only uses the standard library so it can run before the environment exists:

    python scripts/download_amazon.py [--split training|evaluation|all] [--dest data/raw/almrrc2021]

Files already present with the expected size are skipped, so the script can be re-run safely.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET_URL = "https://amazon-last-mile-challenges.s3.amazonaws.com"
PREFIX = "almrrc2021/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects() -> list[tuple[str, int]]:
    url = f"{BUCKET_URL}/?list-type=2&prefix={PREFIX}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    objects = []
    for item in root.findall("s3:Contents", NS):
        key = item.findtext("s3:Key", namespaces=NS) or ""
        size = int(item.findtext("s3:Size", default="0", namespaces=NS))
        if not key.endswith("/"):
            objects.append((key, size))
    return objects


def wanted(key: str, split: str) -> bool:
    if split == "all":
        return True
    return f"almrrc2021-data-{split}/" in key or key.count("/") == 1


def download(key: str, size: int, dest_root: Path) -> None:
    target = dest_root / key.removeprefix(PREFIX)
    if target.exists() and target.stat().st_size == size:
        print(f"skip  {key}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"get   {key} ({size / 1e6:.1f} MB)", flush=True)
    with urllib.request.urlopen(f"{BUCKET_URL}/{key}", timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=8 * 1024 * 1024)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {key}")
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["training", "evaluation", "all"], default="all")
    parser.add_argument("--dest", type=Path, default=Path("data/raw/almrrc2021"))
    args = parser.parse_args()

    objects = [(k, s) for k, s in list_objects() if wanted(k, args.split)]
    total = sum(s for _, s in objects)
    print(f"{len(objects)} files, {total / 1e9:.2f} GB -> {args.dest}")
    for key, size in sorted(objects, key=lambda o: o[1]):
        download(key, size, args.dest)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
