"""Fetch true quantum random seeds from the ANU QRNG API.

The API returns uint16 values. Four consecutive uint16 values are
concatenated (bitwise) to form one 64-bit seed, matching the bit-width
of the LCG-generated PRNG seeds.

Raw API responses are logged to raw_qrng_logs.json for auditability.
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config.settings import (
    NUM_SEEDS,
    QRNG_API_URL,
    QRNG_MAX_PER_CALL,
    QRNG_RETRY_ATTEMPTS,
    QRNG_RETRY_BACKOFF,
    SEEDS_DIR,
)


def fetch_uint16_block(n: int) -> list[int]:
    """Fetch *n* uint16 values from the ANU QRNG API with retry logic."""
    url = f"{QRNG_API_URL}?length={n}&type=uint16"
    backoff = QRNG_RETRY_BACKOFF

    for attempt in range(1, QRNG_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success"):
                raise RuntimeError(f"API returned success=false: {data}")
            return data["data"]
        except (requests.RequestException, RuntimeError) as e:
            print(f"  Attempt {attempt}/{QRNG_RETRY_ATTEMPTS} failed: {e}")
            if attempt < QRNG_RETRY_ATTEMPTS:
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


def fetch_qrng_seeds(n: int = NUM_SEEDS) -> tuple[list[int], list[dict]]:
    """Fetch *n* 64-bit quantum seeds.

    Returns:
        seeds: list of n 64-bit integers
        raw_logs: list of raw API response dicts (for auditability)
    """
    needed_uint16 = n * 4  # 4 uint16 per 64-bit seed
    all_values: list[int] = []
    raw_logs: list[dict] = []

    while len(all_values) < needed_uint16:
        batch_size = min(QRNG_MAX_PER_CALL, needed_uint16 - len(all_values))
        print(f"  Fetching {batch_size} uint16 values from ANU API...")
        values = fetch_uint16_block(batch_size)
        all_values.extend(values)
        raw_logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requested": batch_size,
            "received": len(values),
            "data": values,
        })

    # Concatenate 4 uint16 -> 1 uint64
    seeds: list[int] = []
    for i in range(n):
        base = i * 4
        v0, v1, v2, v3 = all_values[base : base + 4]
        seed_64 = (v0 << 48) | (v1 << 32) | (v2 << 16) | v3
        seeds.append(seed_64)

    return seeds, raw_logs


def save_seeds(seeds: list[int], path: Path) -> None:
    """Write seeds to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "seed"])
        for i, seed in enumerate(seeds):
            writer.writerow([i, seed])


def save_raw_logs(logs: list[dict], path: Path) -> None:
    """Write raw API responses to JSON for auditability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(logs, f, indent=2)


def main() -> None:
    seeds, raw_logs = fetch_qrng_seeds()

    seed_path = SEEDS_DIR / "qrng_seeds.csv"
    log_path = SEEDS_DIR / "raw_qrng_logs.json"

    save_seeds(seeds, seed_path)
    save_raw_logs(raw_logs, log_path)

    print(f"Generated {len(seeds)} QRNG seeds -> {seed_path}")
    print(f"Raw API logs -> {log_path}")
    print(f"  First 5: {seeds[:5]}")


if __name__ == "__main__":
    main()
