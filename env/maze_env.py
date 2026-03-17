"""Custom Gymnasium environment for maze navigation.

Observation (per frame):
  - 5x5 local grid centered on agent (25 values): 0=passage, 1=wall
  - 2 values: normalized goal direction (dx, dy)
  - 1 value: step progress (steps / max_steps)
  - 1 value: whether current cell was already visited (0 or 1)
  Total per frame: 29

With frame stacking (default 4): 29 * 4 = 116D
Frame stacking gives the MLP temporal context — it can infer movement
direction and detect backtracking without needing recurrence.

BFS distance is used ONLY for reward shaping, never in the observation.

Action space: Discrete(4) — Up, Down, Left, Right
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from config.settings import (
    FRAME_STACK,
    GOAL_POS,
    MAX_STEPS,
    REWARD_COLLISION,
    REWARD_GOAL,
    REWARD_REVISIT,
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

VIEW_RADIUS = 2  # 5x5 local view
VIEW_SIZE = 2 * VIEW_RADIUS + 1  # 5
SINGLE_OBS_SIZE = VIEW_SIZE * VIEW_SIZE + 2 + 1 + 1  # 25 + 2 + 1 + 1 = 29


class MazeEnv(gym.Env):
    """Maze navigation with local grid-view + frame stacking."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, mazes: list[np.ndarray], render_mode=None,
                 start=None, goal=None, max_steps=None, n_stack=None):
        super().__init__()

        self.mazes = mazes
        self.render_mode = render_mode
        self.maze_index = 0
        self._start = start or START_POS
        self._goal = goal or GOAL_POS
        self.max_steps = max_steps or MAX_STEPS
        self.n_stack = n_stack or FRAME_STACK

        self.grid_size = mazes[0].shape[0]

        # Stacked observation
        obs_size = SINGLE_OBS_SIZE * self.n_stack
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Precompute BFS distance maps for reward shaping ONLY
        self.distance_maps = [self._bfs(m, self._goal) for m in mazes]

        # State
        self.grid = None
        self.dist_map = None
        self.agent_pos = None
        self.steps = 0
        self.visited = set()
        self._prev_dist = 0
        self._frame_buffer = deque(maxlen=self.n_stack)

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
        self.visited = {self._start}
        self._prev_dist = self.dist_map[self._start[0], self._start[1]]

        # Fill frame buffer with initial observation
        first_frame = self._get_single_obs()
        self._frame_buffer.clear()
        for _ in range(self.n_stack):
            self._frame_buffer.append(first_frame)

        return self._get_stacked_obs(), {}

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

        # ── Reward ──
        reward = REWARD_STEP

        if collision:
            reward += REWARD_COLLISION
        elif reached_goal:
            reward += REWARD_GOAL
        else:
            # BFS-based potential shaping (in reward only, not observation)
            curr_dist = self.dist_map[self.agent_pos[0], self.agent_pos[1]]
            if curr_dist >= 0:
                reward += float(self._prev_dist - curr_dist)
                self._prev_dist = curr_dist

            # Revisit penalty
            if self.agent_pos in self.visited:
                reward += REWARD_REVISIT
            else:
                self.visited.add(self.agent_pos)

        if timed_out and not reached_goal:
            reward += REWARD_TIMEOUT

        terminated = reached_goal
        truncated = timed_out and not reached_goal
        info = {"success": reached_goal, "steps": self.steps}

        # Update frame buffer
        self._frame_buffer.append(self._get_single_obs())

        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_single_obs(self) -> np.ndarray:
        """29D single frame: 5x5 grid + goal dir + progress + visited flag."""
        r, c = self.agent_pos
        gs = self.grid_size

        # 5x5 local view (25 values)
        view = np.ones(VIEW_SIZE * VIEW_SIZE, dtype=np.float32)
        idx = 0
        for dr in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for dc in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                vr, vc = r + dr, c + dc
                if 0 <= vr < gs and 0 <= vc < gs:
                    view[idx] = float(self.grid[vr, vc])
                idx += 1

        # Goal direction normalized to [-1, 1]
        goal_dir = np.array([
            (self._goal[1] - c) / gs,
            (self._goal[0] - r) / gs,
        ], dtype=np.float32)

        # Step progress [0, 1]
        progress = np.float32(self.steps / self.max_steps)

        # Whether current cell was already visited
        revisiting = np.float32(1.0 if self.agent_pos in self.visited else 0.0)

        return np.concatenate([view, goal_dir, [progress, revisiting]])

    def _get_stacked_obs(self) -> np.ndarray:
        """Stack last N frames into single observation vector."""
        return np.concatenate(list(self._frame_buffer))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_rgb(self) -> np.ndarray:
        cell_size = 12
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
