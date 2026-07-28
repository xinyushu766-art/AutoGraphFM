"""Download the model-level China automobile sales CSV with audited fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "felixzhao/china-automobile-monthly-sales-data-2018-2024-4"
)
GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/hoangphuc-ngo/"
    "Forecasting-China-EV-Sales/69a4a61284b301b66c7a31c1de813b476220b55c/"
    "Data/Raw/China%20Automobile%20Sales%20Data.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.destination.exists() and not args.overwrite:
        raise FileExistsError(args.destination)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    attempts = []
    for url in (KAGGLE_URL, GITHUB_RAW_URL):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = response.read()
            if not payload.startswith(b"model,units_sold,make,"):
                raise ValueError("Downloaded payload is not the expected CSV.")
            args.destination.write_bytes(payload)
            manifest = {
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "source_url": url,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "attempts": attempts,
            }
            args.destination.with_suffix(".download.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print(args.destination)
            return
        except Exception as exc:  # noqa: BLE001 - each endpoint is audited
            attempts.append({"url": url, "error": repr(exc)})
    raise RuntimeError(json.dumps({"download_failed": attempts}, indent=2))


if __name__ == "__main__":
    main()
