"""Download, validate, and profile the official SRNI-CAR Sales CSV.

Some managed or corporate networks block Figshare private-link downloads. When
that happens, download the Sales CSV from the official project page in a browser
and validate it with ``scripts/validate_srni_car_file.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from autogfm.realdata import (
    SRNI_CAR_PROJECT_URL,
    SRNI_CAR_SALES_DOWNLOAD_URLS,
    inspect_srni_car_sales,
)


def _looks_like_csv(path: Path) -> None:
    raw = path.read_bytes()[:8192]
    lowered = raw.lower().lstrip()
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html"):
        raise RuntimeError("Download returned HTML instead of the SRNI-CAR CSV.")
    text = ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    required = ("car series", "brand", "year", "month", "sales")
    header = text.splitlines()[0].lower() if text else ""
    if not all(token in header for token in required):
        raise RuntimeError(
            "Downloaded file does not contain the expected SRNI-CAR Sales header."
        )


def _download(url: str, partial: Path, timeout: int) -> int:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 AutoGraphFM/0.12.0",
            "Referer": SRNI_CAR_PROJECT_URL,
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    with (
        urlopen(request, timeout=timeout) as response,
        partial.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    _looks_like_csv(partial)
    return partial.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--url",
        action="append",
        help="Optional download URL. Repeat to try multiple mirrors.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
        help="Directory for download attempts and the authenticity report.",
    )
    args = parser.parse_args()

    destination = args.destination
    validation_dir = args.validation_dir
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Destination exists: {destination}")
    if validation_dir.exists() and any(validation_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Validation directory is not empty: {validation_dir}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)

    urls = tuple(args.url) if args.url else SRNI_CAR_SALES_DOWNLOAD_URLS
    attempts: list[dict[str, object]] = []
    last_error: Exception | None = None
    for url in urls:
        partial.unlink(missing_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        try:
            bytes_written = _download(url, partial, args.timeout)
            attempts.append(
                {
                    "url": url,
                    "started_at": started,
                    "status": "downloaded",
                    "bytes": bytes_written,
                }
            )
            partial.replace(destination)
            last_error = None
            break
        except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
            partial.unlink(missing_ok=True)
            last_error = exc
            attempts.append(
                {
                    "url": url,
                    "started_at": started,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    attempts_path = validation_dir / "download_attempts.json"
    attempts_path.write_text(
        json.dumps(attempts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if last_error is not None or not destination.is_file():
        raise RuntimeError(
            "Automatic SRNI-CAR download failed for every configured Figshare "
            "endpoint. Open the official project page, download the Sales CSV "
            "manually, save it at the requested destination, then run "
            "scripts/validate_srni_car_file.py. "
            f"Attempts: {attempts_path}. Last cause: {last_error}"
        ) from last_error

    inspection = inspect_srni_car_sales(
        destination,
        validation_dir / "authenticity",
        overwrite=args.overwrite,
    )
    print(f"downloaded={destination}")
    print(f"bytes={destination.stat().st_size}")
    print(f"evidence_scope={inspection.evidence_scope}")
    print(f"verification_level={inspection.verification_level}")
    print(
        "authenticity_report="
        f"{validation_dir / 'authenticity' / 'SRNI_AUTHENTICITY_REPORT.md'}"
    )


if __name__ == "__main__":
    main()
