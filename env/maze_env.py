"""Custom Gymnasium environment for maze navigation.

Observation: local grid view + goal direction + step progress.
  - 7x7 grid centered on agent (49 values): wall=1, passage=0, out-of-bounds=1
  - 2 values: normalized (dx, dy) direction to goal
  - 1 value: step progress (steps_taken / max_steps)
  Total: 52D

The agent sees local spatial structure and must learn navigation
strategies from training experience. BFS-based reward shaping guides
learning without trivializing the observation.

Action space: Discrete(4) — Up, Down, Left, Right
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from config.settings import (
    GOAL_POS,
    MAX_STEPS,
    REWARD_COLLISION,
    REWARD_GOAL,
    REWARD_STEP,
    REWARD_TIMEOUT,
    START_POS,
)

ACTION_DELTAS = {
    0: (-1, 0),   # Up
    1: (1, 0),    # Down
    2: (0, -1),   # Left
    3: (0, 1),    # Right
}

VIEW_RADIUS = 3  # 7x7 local view
VIEW_SIZE = 2 * VIEW_RADIUS + 1  # 7
OBS_SIZE = VIEW_SIZE * VIEW_SIZE + 3  # 49 + 2 (goal dir) + 1 (progress) = 52


class MazeEnv(gym.Env):
    """Maze navigation with local grid-view observation."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, mazes: list[np.ndarray], render_mode=None,
                 start=None, goal=None, max_steps=None):
        super().__init__()

        self.mazes = mazes
        self.render_mode = render_mode
        self.maze_index = 0
        self._start = start or START_POS
        self._goal = goal or GOAL_POS
        self.max_steps = max_steps or MAX_STEPS

        self.grid_size = mazes[0].shape[0]

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Precompute BFS distance maps for reward shaping only
        self.distance_maps = [self._bfs(m, self._goal) for m in mazes]

        # State
        self.grid = None
        self.dist_map = None
        self.agent_pos = None
        self.steps = 0
        self._prev_dist = 0

    @staticmethod
    def _bfs(grid: np.ndarray, goal: tuple) -> np.ndarray:
        rows, cols = grid.shape
        dist = np.full((rows, cols), -1, dtype=np.int32)
        dist[goal[0], goal[1]] = 0
        q = deque([goal])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and dist[nr, nc] == -1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.maze_index % len(self.mazes)
        self.grid = self.mazes[idx]
        self.dist_map = self.distance_maps[idx]
        self.maze_index += 1
        self.agent_pos = self._start
        self.steps = 0
        self._prev_dist = self.dist_map[self.agent_pos[0], self.agent_pos[1]]
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        dr, dc = ACTION_DELTAS[int(action)]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        collision = False
        if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                and self.grid[nr, nc] == 0):
            self.agent_pos = (nr, nc)
        else:
            collision = True

        reached_goal = self.agent_pos == self._goal
        timed_out = self.steps >= self.max_steps

        # Reward: BFS-based shaping (in reward, NOT in observation)
        reward = REWARD_STEP
        if collision:
            reward += REWARD_COLLISION
        elif reached_goal:
            reward += REWARD_GOAL
        else:
            curr_dist = self.dist_map[self.agent_pos[0], self.agent_pos[1]]
            if curr_dist >= 0:
                reward += float(self._prev_dist - curr_dist)
                self._prev_dist = curr_dist

        if timed_out and not reached_goal:
            reward += REWARD_TIMEOUT

        terminated = reached_goal
        truncated = timed_out and not reached_goal
        info = {"success": reached_goal, "steps": self.steps}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """52D: 7x7 local grid + goal direction + step progress."""
        r, c = self.agent_pos
        gs = self.grid_size

        # 7x7 local view centered on agent
        view = np.ones(VIEW_SIZE * VIEW_SIZE, dtype=np.float32)  # walls by default
        idx = 0
        for dr in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for dc in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                vr, vc = r + dr, c + dc
                if 0 <= vr < gs and 0 <= vc < gs:
                    view[idx] = float(self.grid[vr, vc])  # 0=passage, 1=wall
                # else: stays 1.0 (out-of-bounds = wall)
                idx += 1

        # Goal direction (normalized to [-1, 1])
        dx = self._goal[1] - c
        dy = self._goal[0] - r
        max_dist = self.grid_size
        goal_dir = np.array([dx / max_dist, dy / max_dist], dtype=np.float32)

        # Step progress
        progress = np.array([self.steps / self.max_steps], dtype=np.float32)

        return np.concatenate([view, goal_dir, progress])

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_rgb(self) -> np.ndarray:
        cell_size = 8
        gs = self.grid_size
        img = np.zeros((gs * cell_size, gs * cell_size, 3), dtype=np.uint8)
        for r in range(gs):
            for c in range(gs):
                color = (0, 0, 0) if self.grid[r, c] == 1 else (255, 255, 255)
                img[r * cell_size:(r + 1) * cell_size,
                    c * cell_size:(c + 1) * cell_size] = color
        ar, ac = self.agent_pos
        img[ar * cell_size:(ar + 1) * cell_size,
            ac * cell_size:(ac + 1) * cell_size] = (0, 0, 255)
        gr, gc = self._goal
        img[gr * cell_size:(gr + 1) * cell_size,
            gc * cell_size:(gc + 1) * cell_size] = (0, 255, 0)
        return img
