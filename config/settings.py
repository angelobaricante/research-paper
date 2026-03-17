"""Central configuration for the research pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEEDS_DIR = DATA_DIR / "seeds"
MAZES_DIR = DATA_DIR / "mazes"
METRICS_DIR = DATA_DIR / "metrics"
MODELS_DIR = DATA_DIR / "models"

# ── Seed Generation ───────────────────────────────────────────────────────
NUM_SEEDS = 100
TRAIN_RATIO = 0.5
NUM_TRAIN = int(NUM_SEEDS * TRAIN_RATIO)   # 50
NUM_TEST = NUM_SEEDS - NUM_TRAIN            # 50

# LCG constants (Numerical Recipes)
LCG_A = 1664525
LCG_C = 1013904223
LCG_M = 2**32
LCG_INITIAL_STATE = 12345  # fixed for reproducibility

# ANU QRNG API
QRNG_API_URL = "https://qrng.anu.edu.au/API/jsonI.php"
QRNG_MAX_PER_CALL = 1024
QRNG_RETRY_ATTEMPTS = 3
QRNG_RETRY_BACKOFF = 2.0  # seconds, multiplied each retry

# ── Maze Generation ───────────────────────────────────────────────────────
MAZE_CELLS = 7                            # 7x7 logical cells
MAZE_GRID_SIZE = 2 * MAZE_CELLS + 1      # 15x15 actual grid
START_POS = (1, 1)
GOAL_POS = (MAZE_GRID_SIZE - 2, MAZE_GRID_SIZE - 2)  # (13, 13)

# ── Gymnasium Environment ─────────────────────────────────────────────────
MAX_STEPS = 400                           # ~5x mean optimal for 7x7
FRAME_STACK = 4                           # stack last 4 observations

# Rewards
REWARD_GOAL = 100.0
REWARD_STEP = -0.1
REWARD_COLLISION = -0.2
REWARD_TIMEOUT = -5.0
REWARD_REVISIT = -0.3

# ── A2C Hyperparameters ──────────────────────────────────────────────────
A2C_CONFIG = {
    "learning_rate": 7e-4,
    "n_steps": 256,
    "ent_coef": 0.02,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [128, 128]},
}
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 100_000
RL_SEED = 0  # fixed RL seed so only maze distribution differs

# ── Statistical Testing ──────────────────────────────────────────────────
ALPHA = 0.05
PERMANOVA_PERMUTATIONS = 1000
