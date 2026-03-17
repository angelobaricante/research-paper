"""Validate seed quality: Shannon entropy and autocorrelation.

Ensures both PRNG (LCG) and QRNG seed sets meet randomness standards
and that bitwise concatenation did not introduce artificial correlations.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.settings import SEEDS_DIR


def load_seeds(path: Path) -> np.ndarray:
    """Load seeds from CSV, return as numpy array of int64."""
    seeds = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seeds.append(int(row["seed"]))
    return np.array(seeds, dtype=np.uint64)


def shannon_entropy_bits(seeds: np.ndarray) -> float:
    """Compute Shannon entropy over the byte-level distribution of all seeds.

    Each 64-bit seed is split into 8 bytes. Entropy is computed over the
    distribution of all byte values (0-255). Maximum entropy = 8.0 bits.
    """
    raw_bytes = np.array(seeds, dtype=np.uint64).view(np.uint8)
    counts = Counter(raw_bytes.tolist())
    total = len(raw_bytes)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def autocorrelation(seeds: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute autocorrelation of the seed sequence at lags 1..max_lag."""
    x = seeds.astype(np.float64)
    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag)
    n = len(x)
    acorrs = []
    for lag in range(1, max_lag + 1):
        c = np.sum(x[: n - lag] * x[lag:]) / ((n - lag) * var)
        acorrs.append(c)
    return np.array(acorrs)


def validate_and_plot(prng_path: Path, qrng_path: Path) -> None:
    """Run validation on both seed sets and produce comparison plots."""
    prng = load_seeds(prng_path)
    qrng = load_seeds(qrng_path)

    # Shannon entropy
    h_prng = shannon_entropy_bits(prng)
    h_qrng = shannon_entropy_bits(qrng)
    print(f"Shannon Entropy (byte-level, max=8.0 bits):")
    print(f"  PRNG (LCG): {h_prng:.4f}")
    print(f"  QRNG:       {h_qrng:.4f}")

    # Autocorrelation
    ac_prng = autocorrelation(prng)
    ac_qrng = autocorrelation(qrng)
    print(f"\nAutocorrelation (lag 1): PRNG={ac_prng[0]:.4f}, QRNG={ac_qrng[0]:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Entropy bar chart
    axes[0].bar(["PRNG (LCG)", "QRNG"], [h_prng, h_qrng], color=["#e74c3c", "#2ecc71"])
    axes[0].axhline(y=8.0, color="gray", linestyle="--", label="Max (8.0)")
    axes[0].set_ylabel("Shannon Entropy (bits)")
    axes[0].set_title("Byte-Level Shannon Entropy")
    axes[0].legend()
    axes[0].set_ylim(0, 8.5)

    # Autocorrelation
    lags = np.arange(1, len(ac_prng) + 1)
    axes[1].stem(lags - 0.15, ac_prng, linefmt="r-", markerfmt="ro", basefmt="r-", label="PRNG (LCG)")
    axes[1].stem(lags + 0.15, ac_qrng, linefmt="g-", markerfmt="go", basefmt="g-", label="QRNG")
    axes[1].axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ci = 1.96 / np.sqrt(len(prng))
    axes[1].axhline(y=ci, color="blue", linestyle="--", alpha=0.5, label="95% CI")
    axes[1].axhline(y=-ci, color="blue", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].set_title("Seed Sequence Autocorrelation")
    axes[1].legend()

    plt.tight_layout()
    out_path = SEEDS_DIR / "seed_validation.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out_path}")


def main() -> None:
    prng_path = SEEDS_DIR / "prng_seeds.csv"
    qrng_path = SEEDS_DIR / "qrng_seeds.csv"

    if not prng_path.exists() or not qrng_path.exists():
        print("Error: seed CSVs not found. Run generate_prng_seeds.py and fetch_qrng_seeds.py first.")
        return

    validate_and_plot(prng_path, qrng_path)


if __name__ == "__main__":
    main()
