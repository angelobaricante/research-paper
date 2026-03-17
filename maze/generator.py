"""Recursive backtracking maze generator (iterative stack implementation).

Produces a perfect maze (exactly one path between any two cells) on a
(2*size+1) x (2*size+1) grid where 1=wall, 0=passage.
"""

import csv
from pathlib import Path

import numpy as np

from config.settings import MAZE_CELLS, MAZE_GRID_SIZE, SEEDS_DIR, MAZES_DIR


def generate_maze(size: int = MAZE_CELLS, seed: int = 0) -> np.ndarray:
    """Generate a perfect maze using iterative recursive backtracking.

    Args:
        size: Number of cells per side (e.g. 20 -> 41x41 grid).
        seed: 64-bit integer seed for the RNG used to shuffle neighbors.

    Returns:
        grid: numpy array of shape (2*size+1, 2*size+1), dtype=uint8.
              0 = passage, 1 = wall.
    """
    grid_size = 2 * size + 1
    grid = np.ones((grid_size, grid_size), dtype=np.uint8)
    rng = np.random.default_rng(seed)

    # Directions: (row_delta, col_delta) for N, S, E, W
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    # Start at cell (0, 0) -> grid position (1, 1)
    start_r, start_c = 0, 0
    grid[1, 1] = 0
    visited = np.zeros((size, size), dtype=bool)
    visited[start_r, start_c] = True

    stack = [(start_r, start_c)]

    while stack:
        cr, cc = stack[-1]

        # Find unvisited neighbors
        neighbors = []
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < size and 0 <= nc < size and not visited[nr, nc]:
                neighbors.append((nr, nc, dr, dc))

        if neighbors:
            # Pick a random unvisited neighbor
            idx = rng.integers(len(neighbors))
            nr, nc, dr, dc = neighbors[idx]

            # Carve passage: remove wall between current and neighbor
            wall_r = 1 + 2 * cr + dr
            wall_c = 1 + 2 * cc + dc
            grid[wall_r, wall_c] = 0

            # Carve the neighbor cell
            grid[1 + 2 * nr, 1 + 2 * nc] = 0

            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


def load_seeds(path: Path) -> list[int]:
    """Load seeds from a CSV file."""
    seeds = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seeds.append(int(row["seed"]))
    return seeds


def generate_all_mazes(seed_type: str) -> np.ndarray:
    """Generate all mazes for a seed type ('prng' or 'qrng').

    Returns:
        mazes: numpy array of shape (n_seeds, grid_size, grid_size)
    """
    csv_path = SEEDS_DIR / f"{seed_type}_seeds.csv"
    seeds = load_seeds(csv_path)
    mazes = np.stack([generate_maze(MAZE_CELLS, s) for s in seeds])
    return mazes


def save_mazes(mazes: np.ndarray, seed_type: str) -> Path:
    """Save mazes as a compressed .npz file."""
    MAZES_DIR.mkdir(parents=True, exist_ok=True)
    path = MAZES_DIR / f"{seed_type}_mazes.npz"
    np.savez_compressed(path, mazes=mazes)
    print(f"Saved {mazes.shape[0]} mazes -> {path}")
    return path


def load_mazes(seed_type: str) -> np.ndarray:
    """Load mazes from .npz file."""
    path = MAZES_DIR / f"{seed_type}_mazes.npz"
    return np.load(path)["mazes"]


def main() -> None:
    for seed_type in ["prng", "qrng"]:
        csv_path = SEEDS_DIR / f"{seed_type}_seeds.csv"
        if not csv_path.exists():
            print(f"Skipping {seed_type}: {csv_path} not found")
            continue
        mazes = generate_all_mazes(seed_type)
        save_mazes(mazes, seed_type)


if __name__ == "__main__":
    main()
