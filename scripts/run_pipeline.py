"""End-to-end pipeline: seeds -> mazes -> metrics -> training -> evaluation -> analysis.

Usage:
    python -m scripts.run_pipeline [--skip-qrng-fetch]

Use --skip-qrng-fetch if QRNG seeds are already saved (avoids API calls).
"""

import argparse
from pathlib import Path

import pandas as pd

from config.settings import METRICS_DIR, SEEDS_DIR, MAZES_DIR, MODELS_DIR


def step_1_generate_seeds(skip_qrng_fetch: bool = False) -> None:
    """Generate PRNG (LCG) seeds and fetch QRNG seeds."""
    print("\n" + "=" * 60)
    print("STEP 1: Seed Generation")
    print("=" * 60)

    from seeds.generate_prng_seeds import main as generate_prng
    generate_prng()

    if skip_qrng_fetch:
        if (SEEDS_DIR / "qrng_seeds.csv").exists():
            print("Skipping QRNG fetch (--skip-qrng-fetch, file exists)")
        else:
            print("WARNING: --skip-qrng-fetch but qrng_seeds.csv not found!")
            print("  Run 'python3 -m seeds.fetch_qrng_seeds' when you have internet access.")
            print("  Pipeline will continue with PRNG-only steps.")
    else:
        from seeds.fetch_qrng_seeds import main as fetch_qrng
        fetch_qrng()


def step_2_validate_seeds() -> None:
    """Run seed validation (entropy + autocorrelation)."""
    print("\n" + "=" * 60)
    print("STEP 2: Seed Validation")
    print("=" * 60)

    prng_exists = (SEEDS_DIR / "prng_seeds.csv").exists()
    qrng_exists = (SEEDS_DIR / "qrng_seeds.csv").exists()

    if prng_exists and qrng_exists:
        from seeds.validate_seeds import main as validate
        validate()
    else:
        missing = []
        if not prng_exists:
            missing.append("prng_seeds.csv")
        if not qrng_exists:
            missing.append("qrng_seeds.csv")
        print(f"  Skipping validation: missing {', '.join(missing)}")


def step_3_generate_mazes() -> None:
    """Generate mazes for both seed groups."""
    print("\n" + "=" * 60)
    print("STEP 3: Maze Generation")
    print("=" * 60)

    from maze.generator import main as gen_mazes
    gen_mazes()


def step_4_compute_maze_metrics() -> None:
    """Compute structural metrics for all mazes."""
    print("\n" + "=" * 60)
    print("STEP 4: Maze Structural Metrics")
    print("=" * 60)

    import numpy as np
    from maze.generator import load_mazes
    from maze.metrics import compute_metrics_batch
    from maze.era import compute_era_features, plot_era

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for seed_type in ["prng", "qrng"]:
        maze_path = MAZES_DIR / f"{seed_type}_mazes.npz"
        if not maze_path.exists():
            print(f"\n  Skipping {seed_type.upper()}: {maze_path} not found")
            continue
        mazes = load_mazes(seed_type)
        print(f"\n  Computing metrics for {seed_type.upper()} ({mazes.shape[0]} mazes)...")

        metrics = compute_metrics_batch(mazes)
        df = pd.DataFrame(metrics)
        out_path = METRICS_DIR / f"{seed_type}_maze_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")
        print(f"  Sample:\n{df.describe().to_string()}")

    # ERA analysis (only if both groups exist)
    prng_maze_path = MAZES_DIR / "prng_mazes.npz"
    qrng_maze_path = MAZES_DIR / "qrng_mazes.npz"
    if prng_maze_path.exists() and qrng_maze_path.exists():
        print("\n  Computing Expressive Range Analysis...")
        prng_mazes = load_mazes("prng")
        qrng_mazes = load_mazes("qrng")
        prng_era = compute_era_features(prng_mazes)
        qrng_era = compute_era_features(qrng_mazes)
        plot_era(prng_era, qrng_era, save_path=METRICS_DIR / "era_analysis.png")
    else:
        print("\n  Skipping ERA: need both PRNG and QRNG mazes")


def _both_metrics_exist() -> bool:
    """Check if both PRNG and QRNG maze metric CSVs exist."""
    prng = (METRICS_DIR / "prng_maze_metrics.csv").exists()
    qrng = (METRICS_DIR / "qrng_maze_metrics.csv").exists()
    if not (prng and qrng):
        missing = []
        if not prng:
            missing.append("prng_maze_metrics.csv")
        if not qrng:
            missing.append("qrng_maze_metrics.csv")
        print(f"  Skipping: missing {', '.join(missing)}")
        print("  Fetch QRNG seeds first, then re-run from the appropriate step.")
    return prng and qrng


def _both_models_exist() -> bool:
    """Check if both trained models exist."""
    prng = (MODELS_DIR / "a2c_prng" / "final_model.zip").exists()
    qrng = (MODELS_DIR / "a2c_qrng" / "final_model.zip").exists()
    if not (prng and qrng):
        missing = []
        if not prng:
            missing.append("a2c_prng")
        if not qrng:
            missing.append("a2c_qrng")
        print(f"  Skipping: missing model(s) {', '.join(missing)}")
    return prng and qrng


def step_5_statistical_tests_mazes() -> None:
    """Run statistical tests on maze structural metrics."""
    print("\n" + "=" * 60)
    print("STEP 5: Statistical Tests (Maze Structure)")
    print("=" * 60)

    if not _both_metrics_exist():
        return

    from analysis.statistical_tests import run_maze_structural_tests, run_manova

    prng_df = pd.read_csv(METRICS_DIR / "prng_maze_metrics.csv")
    qrng_df = pd.read_csv(METRICS_DIR / "qrng_maze_metrics.csv")

    results = run_maze_structural_tests(prng_df, qrng_df)
    results.to_csv(METRICS_DIR / "maze_statistical_tests.csv", index=False)
    print(f"\n  Results saved to {METRICS_DIR / 'maze_statistical_tests.csv'}")

    # MANOVA
    manova_cols = ["path_length", "tortuosity", "dead_end_count",
                   "junction_3_proportion", "junction_4_proportion"]
    run_manova(prng_df, qrng_df, manova_cols)


def step_6_train_agents() -> None:
    """Train A2C agents on both maze groups."""
    print("\n" + "=" * 60)
    print("STEP 6: A2C Training")
    print("=" * 60)

    from training.train import train_a2c

    for seed_type in ["prng", "qrng"]:
        maze_path = MAZES_DIR / f"{seed_type}_mazes.npz"
        if not maze_path.exists():
            print(f"  Skipping {seed_type.upper()}: {maze_path} not found")
            continue
        train_a2c(seed_type)


def step_7_evaluate_agents() -> None:
    """Evaluate agents on unseen mazes (intra + cross domain)."""
    print("\n" + "=" * 60)
    print("STEP 7: Agent Evaluation")
    print("=" * 60)

    if not _both_models_exist():
        return

    from evaluation.evaluate import run_all_evaluations
    run_all_evaluations()


def step_8_statistical_tests_rl() -> None:
    """Run statistical tests on RL evaluation results."""
    print("\n" + "=" * 60)
    print("STEP 8: Statistical Tests (RL Performance)")
    print("=" * 60)

    required = [
        METRICS_DIR / "eval_a2c_prng_on_prng.csv",
        METRICS_DIR / "eval_a2c_qrng_on_qrng.csv",
        METRICS_DIR / "a2c_prng_training_log.csv",
        METRICS_DIR / "a2c_qrng_training_log.csv",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        print(f"  Skipping: missing {', '.join(missing)}")
        return

    from analysis.statistical_tests import run_rl_evaluation_tests
    from analysis.plots import plot_training_curves, plot_evaluation_summary

    # Compare intra-domain performance
    prng_eval = pd.read_csv(METRICS_DIR / "eval_a2c_prng_on_prng.csv")
    qrng_eval = pd.read_csv(METRICS_DIR / "eval_a2c_qrng_on_qrng.csv")
    results = run_rl_evaluation_tests(prng_eval, qrng_eval)
    results.to_csv(METRICS_DIR / "rl_statistical_tests.csv", index=False)

    # Training curves
    plot_training_curves(
        METRICS_DIR / "a2c_prng_training_log.csv",
        METRICS_DIR / "a2c_qrng_training_log.csv",
    )

    # Evaluation summary
    plot_evaluation_summary()

    print(f"\n  All results saved to {METRICS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Run full research pipeline")
    parser.add_argument(
        "--skip-qrng-fetch",
        action="store_true",
        help="Skip QRNG API fetch (use existing qrng_seeds.csv)",
    )
    parser.add_argument(
        "--start-from",
        type=int, default=1, choices=range(1, 9),
        help="Start from step N (1-8). Useful for resuming.",
    )
    args = parser.parse_args()

    steps = [
        (1, lambda: step_1_generate_seeds(args.skip_qrng_fetch)),
        (2, step_2_validate_seeds),
        (3, step_3_generate_mazes),
        (4, step_4_compute_maze_metrics),
        (5, step_5_statistical_tests_mazes),
        (6, step_6_train_agents),
        (7, step_7_evaluate_agents),
        (8, step_8_statistical_tests_rl),
    ]

    for step_num, step_fn in steps:
        if step_num >= args.start_from:
            step_fn()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
