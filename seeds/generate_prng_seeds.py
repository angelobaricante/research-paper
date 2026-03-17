"""Generate 64-bit seeds using a Linear Congruential Generator (LCG).

Uses the Numerical Recipes constants:
    X_{n+1} = (1664525 * X_n + 1013904223) mod 2^32

Each 64-bit seed is built by concatenating two consecutive 32-bit LCG outputs.
"""

import csv
from pathlib import Path

from config.settings import (
    LCG_A,
    LCG_C,
    LCG_INITIAL_STATE,
    LCG_M,
    NUM_SEEDS,
    SEEDS_DIR,
)


def lcg_next(state: int) -> int:
    """Produce the next LCG state."""
    return (LCG_A * state + LCG_C) % LCG_M


def generate_lcg_seeds(n: int = NUM_SEEDS, x0: int = LCG_INITIAL_STATE) -> list[int]:
    """Generate *n* 64-bit seeds from two consecutive 32-bit LCG outputs.

    Returns a list of n 64-bit integers.
    """
    seeds: list[int] = []
    state = x0
    for _ in range(n):
        state = lcg_next(state)
        high = state
        state = lcg_next(state)
        low = state
        seed_64 = (high << 32) | low
        seeds.append(seed_64)
    return seeds


def save_seeds(seeds: list[int], path: Path) -> None:
    """Write seeds to a CSV file with columns [index, seed]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "seed"])
        for i, seed in enumerate(seeds):
            writer.writerow([i, seed])


def main() -> None:
    seeds = generate_lcg_seeds()
    out_path = SEEDS_DIR / "prng_seeds.csv"
    save_seeds(seeds, out_path)
    print(f"Generated {len(seeds)} LCG seeds -> {out_path}")
    print(f"  First 5: {seeds[:5]}")
    print(f"  Last  5: {seeds[-5:]}")


if __name__ == "__main__":
    main()
