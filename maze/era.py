"""Expressive Range Analysis (ERA) for maze diversity evaluation.

Computes linearity and leniency for each maze, visualizes the ERA scatter
plot, and measures convex hull area for each group.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from maze.graph_utils import build_cell_graph, build_corridor_graph


def compute_linearity(grid: np.ndarray) -> float:
    """Fraction of straight corridors over total corridors.

    Linearity = (corridors with 0 turns) / (total corridors)
    """
    cell_g = build_cell_graph(grid)
    corr_g = build_corridor_graph(cell_g)

    total = corr_g.number_of_edges()
    if total == 0:
        return 0.0
    straight = sum(1 for _, _, d in corr_g.edges(data=True) if d.get("is_straight", False))
    return straight / total


def compute_leniency(grid: np.ndarray) -> float:
    """Negative ratio of dead-ends to total passage cells.

    Leniency = -(dead_end_count) / total_passage_cells
    """
    cell_g = build_cell_graph(grid)
    corr_g = build_corridor_graph(cell_g)

    dead_ends = sum(1 for n in corr_g.nodes() if corr_g.degree(n) == 1)
    total_cells = int(np.sum(grid == 0))
    if total_cells == 0:
        return 0.0
    return -dead_ends / total_cells


def compute_era_features(mazes: np.ndarray) -> np.ndarray:
    """Compute (linearity, leniency) for each maze.

    Returns array of shape (n_mazes, 2).
    """
    features = []
    for i in range(mazes.shape[0]):
        lin = compute_linearity(mazes[i])
        lenien = compute_leniency(mazes[i])
        features.append([lin, lenien])
    return np.array(features)


def convex_hull_area(points: np.ndarray) -> float:
    """Compute convex hull area of 2D points.

    Returns 0.0 if fewer than 3 unique points.
    """
    unique = np.unique(points, axis=0)
    if len(unique) < 3:
        return 0.0
    try:
        hull = ConvexHull(unique)
        return hull.volume  # in 2D, volume = area
    except Exception:
        return 0.0


def plot_era(
    prng_features: np.ndarray,
    qrng_features: np.ndarray,
    save_path=None,
) -> None:
    """Plot ERA scatter with convex hulls for both groups."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter
    ax.scatter(
        prng_features[:, 0], prng_features[:, 1],
        c="red", alpha=0.3, s=15, label="PRNG (LCG)",
    )
    ax.scatter(
        qrng_features[:, 0], qrng_features[:, 1],
        c="green", alpha=0.3, s=15, label="QRNG",
    )

    # Convex hulls
    for features, color, label in [
        (prng_features, "red", "PRNG"),
        (qrng_features, "green", "QRNG"),
    ]:
        unique = np.unique(features, axis=0)
        if len(unique) >= 3:
            hull = ConvexHull(unique)
            area = hull.volume
            vertices = np.append(hull.vertices, hull.vertices[0])
            ax.plot(
                unique[vertices, 0], unique[vertices, 1],
                color=color, linewidth=1.5,
            )
            ax.fill(
                unique[vertices, 0], unique[vertices, 1],
                color=color, alpha=0.08,
            )
            ax.annotate(
                f"{label} hull area: {area:.6f}",
                xy=(0.02, 0.98 if label == "PRNG" else 0.93),
                xycoords="axes fraction",
                fontsize=9, color=color,
                verticalalignment="top",
            )

    ax.set_xlabel("Linearity")
    ax.set_ylabel("Leniency")
    ax.set_title("Expressive Range Analysis")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"ERA plot saved to {save_path}")
    plt.close()
