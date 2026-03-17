"""Structural metrics for maze evaluation."""

import numpy as np
import networkx as nx

from config.settings import START_POS, GOAL_POS
from maze.graph_utils import build_cell_graph, build_corridor_graph


def compute_metrics(grid: np.ndarray) -> dict:
    """Compute all structural metrics for a single maze.

    Returns a dict with keys:
        path_length, manhattan_distance, tortuosity,
        dead_end_count, junction_3_count, junction_4_count,
        junction_3_proportion, junction_4_proportion,
        total_junctions, turn_count, straight_count,
        total_corridors, total_passage_cells
    """
    cell_g = build_cell_graph(grid)
    corr_g = build_corridor_graph(cell_g)

    # Path length (shortest path on cell graph)
    if nx.has_path(cell_g, START_POS, GOAL_POS):
        path_length = nx.shortest_path_length(cell_g, START_POS, GOAL_POS)
    else:
        path_length = float("inf")

    # Manhattan distance
    manhattan = abs(GOAL_POS[0] - START_POS[0]) + abs(GOAL_POS[1] - START_POS[1])

    # Tortuosity
    tortuosity = path_length / manhattan if manhattan > 0 else 1.0

    # Dead-end count (degree-1 in corridor graph)
    dead_end_count = sum(1 for n in corr_g.nodes() if corr_g.degree(n) == 1)

    # Junction counts (degree-3 and degree-4 in corridor graph)
    junction_3 = sum(1 for n in corr_g.nodes() if corr_g.degree(n) == 3)
    junction_4 = sum(1 for n in corr_g.nodes() if corr_g.degree(n) >= 4)
    total_junctions = junction_3 + junction_4

    j3_prop = junction_3 / total_junctions if total_junctions > 0 else 0.0
    j4_prop = junction_4 / total_junctions if total_junctions > 0 else 0.0

    # Turns and straightaways (corridor edges)
    turn_count = 0
    straight_count = 0
    for _, _, data in corr_g.edges(data=True):
        if data.get("is_straight", False):
            straight_count += 1
        else:
            turn_count += 1

    total_corridors = corr_g.number_of_edges()
    total_passage_cells = int(np.sum(grid == 0))

    return {
        "path_length": path_length,
        "manhattan_distance": manhattan,
        "tortuosity": tortuosity,
        "dead_end_count": dead_end_count,
        "junction_3_count": junction_3,
        "junction_4_count": junction_4,
        "junction_3_proportion": j3_prop,
        "junction_4_proportion": j4_prop,
        "total_junctions": total_junctions,
        "turn_count": turn_count,
        "straight_count": straight_count,
        "total_corridors": total_corridors,
        "total_passage_cells": total_passage_cells,
    }


def compute_metrics_batch(mazes: np.ndarray) -> list[dict]:
    """Compute metrics for an array of mazes."""
    return [compute_metrics(mazes[i]) for i in range(mazes.shape[0])]
