"""Graph representations of mazes: cell graph and corridor graph."""

import numpy as np
import networkx as nx


def build_cell_graph(grid: np.ndarray) -> nx.Graph:
    """Build a graph where every passage cell is a node.

    Edges connect orthogonally adjacent passage cells.
    Node IDs are (row, col) tuples.
    """
    rows, cols = grid.shape
    G = nx.Graph()

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0:
                G.add_node((r, c))
                # Check right and down neighbors to avoid duplicate edges
                if c + 1 < cols and grid[r, c + 1] == 0:
                    G.add_edge((r, c), (r, c + 1))
                if r + 1 < rows and grid[r + 1, c] == 0:
                    G.add_edge((r, c), (r + 1, c))

    return G


def build_corridor_graph(cell_graph: nx.Graph) -> nx.Graph:
    """Collapse degree-2 chains into single corridor edges.

    Nodes in the corridor graph are junctions (degree >= 3) and
    dead-ends (degree == 1). Corridor edges store:
        - 'length': number of cells in the collapsed chain
        - 'turns': number of direction changes along the corridor
        - 'is_straight': True if corridor has zero turns
    """
    G = cell_graph
    corridor = nx.Graph()

    # Identify junction and dead-end nodes
    important = {n for n in G.nodes() if G.degree(n) != 2}

    for node in important:
        corridor.add_node(node, degree=G.degree(node))

    # BFS along degree-2 chains from each important node
    visited_edges = set()

    for start in important:
        for neighbor in G.neighbors(start):
            edge = tuple(sorted((start, neighbor)))
            if edge in visited_edges:
                continue

            # Walk along the chain
            path = [start, neighbor]
            visited_edges.add(edge)
            current = neighbor

            while current not in important:
                nexts = [n for n in G.neighbors(current) if n != path[-2]]
                if not nexts:
                    break
                nxt = nexts[0]
                e = tuple(sorted((current, nxt)))
                visited_edges.add(e)
                path.append(nxt)
                current = nxt

            # Compute corridor properties
            end = path[-1]
            length = len(path) - 1  # number of edges = cells - 1
            turns = _count_turns(path)

            if not corridor.has_edge(start, end):
                corridor.add_edge(
                    start, end,
                    length=length,
                    turns=turns,
                    is_straight=(turns == 0),
                )

    return corridor


def _count_turns(path: list[tuple]) -> int:
    """Count direction changes along a path of (row, col) positions."""
    if len(path) < 3:
        return 0
    turns = 0
    prev_dr = path[1][0] - path[0][0]
    prev_dc = path[1][1] - path[0][1]
    for i in range(2, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        if (dr, dc) != (prev_dr, prev_dc):
            turns += 1
        prev_dr, prev_dc = dr, dc
    return turns
