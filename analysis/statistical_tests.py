"""Statistical validation: normality tests, group comparisons, multivariate tests."""

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import ALPHA


def test_normality(data: np.ndarray, label: str = "") -> tuple[float, float, bool]:
    """Shapiro-Wilk normality test.

    Returns (statistic, p_value, is_normal).
    """
    stat, p = stats.shapiro(data)
    is_normal = p > ALPHA
    print(f"  Shapiro-Wilk [{label}]: W={stat:.4f}, p={p:.4f} -> {'Normal' if is_normal else 'Non-normal'}")
    return stat, p, is_normal


def compare_groups(
    prng_data: np.ndarray,
    qrng_data: np.ndarray,
    metric_name: str,
) -> dict:
    """Compare two groups with appropriate test (Welch-t or Mann-Whitney U).

    Automatically selects based on Shapiro-Wilk normality of both groups.
    """
    print(f"\n--- {metric_name} ---")
    print(f"  PRNG: mean={np.mean(prng_data):.4f}, median={np.median(prng_data):.4f}, std={np.std(prng_data):.4f}")
    print(f"  QRNG: mean={np.mean(qrng_data):.4f}, median={np.median(qrng_data):.4f}, std={np.std(qrng_data):.4f}")

    _, p_prng, normal_prng = test_normality(prng_data, f"{metric_name}/PRNG")
    _, p_qrng, normal_qrng = test_normality(qrng_data, f"{metric_name}/QRNG")

    if normal_prng and normal_qrng:
        # Welch's t-test
        stat, p = stats.ttest_ind(prng_data, qrng_data, equal_var=False)
        test_name = "Welch t-test"
        # Cohen's d
        pooled_std = np.sqrt(
            (np.std(prng_data, ddof=1) ** 2 + np.std(qrng_data, ddof=1) ** 2) / 2
        )
        effect_size = (np.mean(prng_data) - np.mean(qrng_data)) / pooled_std if pooled_std > 0 else 0
        effect_name = "Cohen's d"
    else:
        # Mann-Whitney U
        stat, p = stats.mannwhitneyu(prng_data, qrng_data, alternative="two-sided")
        test_name = "Mann-Whitney U"
        # Rank-biserial correlation
        n1, n2 = len(prng_data), len(qrng_data)
        effect_size = 1 - (2 * stat) / (n1 * n2)
        effect_name = "Rank-biserial r"

    significant = p < ALPHA
    print(f"  {test_name}: stat={stat:.4f}, p={p:.6f} -> {'SIGNIFICANT' if significant else 'not significant'}")
    print(f"  {effect_name}: {effect_size:.4f}")

    return {
        "metric": metric_name,
        "test": test_name,
        "statistic": stat,
        "p_value": p,
        "significant": significant,
        "effect_size": effect_size,
        "effect_size_name": effect_name,
        "prng_mean": np.mean(prng_data),
        "qrng_mean": np.mean(qrng_data),
    }


def run_maze_structural_tests(
    prng_metrics: pd.DataFrame,
    qrng_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Run statistical tests on all maze structural metrics."""
    columns_to_test = [
        "path_length", "tortuosity", "dead_end_count",
        "junction_3_proportion", "junction_4_proportion",
        "turn_count", "straight_count",
    ]

    results = []
    for col in columns_to_test:
        if col in prng_metrics.columns and col in qrng_metrics.columns:
            result = compare_groups(
                prng_metrics[col].values,
                qrng_metrics[col].values,
                col,
            )
            results.append(result)

    return pd.DataFrame(results)


def run_rl_evaluation_tests(
    prng_eval: pd.DataFrame,
    qrng_eval: pd.DataFrame,
) -> pd.DataFrame:
    """Run statistical tests on RL evaluation metrics."""
    results = []
    for col in ["success", "reward", "steps"]:
        result = compare_groups(
            prng_eval[col].values,
            qrng_eval[col].values,
            f"RL_{col}",
        )
        results.append(result)
    return pd.DataFrame(results)


def run_manova(
    prng_metrics: pd.DataFrame,
    qrng_metrics: pd.DataFrame,
    columns: list[str],
) -> dict:
    """Run one-way MANOVA on structural metrics.

    Returns dict with Wilks' Lambda, F-value, p-value.
    """
    from statsmodels.multivariate.manova import MANOVA

    # Filter out zero-variance columns (cause singular matrix)
    valid_cols = []
    combined_prng = prng_metrics[columns]
    combined_qrng = qrng_metrics[columns]
    for col in columns:
        if combined_prng[col].std() > 0 and combined_qrng[col].std() > 0:
            valid_cols.append(col)
        else:
            print(f"  Dropping '{col}' from MANOVA (zero variance)")

    if len(valid_cols) < 2:
        print("\n  Skipping MANOVA: need at least 2 variables with non-zero variance")
        return {"manova_result": None}

    prng_subset = prng_metrics[valid_cols].copy()
    qrng_subset = qrng_metrics[valid_cols].copy()
    prng_subset["group"] = "prng"
    qrng_subset["group"] = "qrng"
    combined = pd.concat([prng_subset, qrng_subset], ignore_index=True)

    formula = " + ".join(valid_cols) + " ~ group"
    try:
        manova = MANOVA.from_formula(formula, data=combined)
        result = manova.mv_test()
        print("\n=== MANOVA Results ===")
        print(result.summary())
        return {"manova_result": result}
    except Exception as e:
        print(f"\n  MANOVA failed: {e}")
        return {"manova_result": None}
