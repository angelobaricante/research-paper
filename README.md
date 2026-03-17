# Enhancing Autonomous Navigation Robustness Using Quantum-Seeded Procedural Maze Generation

**A2C Component** — Mark Angelo R. Baricante

Bachelor of Science in Computer Science Thesis
Batangas State University — The National Engineering University
April 2026

## Overview

This repository contains the A2C (Advantage Actor-Critic) pipeline for evaluating the impact of Quantum Random Number Generator (QRNG) seeds compared to Pseudo-Random Number Generator (PRNG) seeds on training Deep Reinforcement Learning agents for autonomous maze navigation.

**Key Finding:** The QRNG-trained A2C agent significantly outperforms the LCG-trained agent across all evaluation metrics (p < 0.000001), demonstrating that seed entropy quality directly impacts DRL agent generalization.

## Results Summary

### Agent Evaluation (on 50 unseen test mazes)

| Metric | AgentPRNG (LCG) | AgentQRNG | p-value | Effect Size (r) |
|---|---|---|---|---|
| Success Rate | 14% | **86%** | < 0.000001 | 0.72 (large) |
| Mean Reward | -73.6 | **+92.6** | < 0.000001 | 0.68 (large) |
| Mean Steps | 351 | **123** | < 0.000001 | 0.70 (large) |

### Generalization Gap (SR_intra - SR_cross)

| Agent | Intra-Domain | Cross-Domain | Gap |
|---|---|---|---|
| AgentPRNG | 14% | 8% | 6% (fragile) |
| AgentQRNG | 86% | 84% | **2% (robust)** |

## Experimental Design

### Seed Sources

- **PRNG (Control):** Linear Congruential Generator (Numerical Recipes)
  - Formula: `X_{n+1} = (1664525 * X_n + 1013904223) mod 2^32`
  - 64-bit seeds built by concatenating two consecutive 32-bit LCG outputs
  - Known weaknesses: sequential correlation, 32-bit state, short effective period
- **QRNG (Experimental):** Australian National University Quantum API
  - True quantum randomness from vacuum fluctuation measurements
  - Four uint16 values concatenated into 64-bit seeds

### Maze Generation

- **Algorithm:** Recursive backtracking (iterative stack implementation)
- **Size:** 7x7 logical cells (15x15 actual grid with walls)
- **Seeds per group:** 100 (50 training / 50 testing)
- **Seeding:** Both seed types fed into `np.random.default_rng(seed)` for maze generation. The independent variable is the **seed distribution**, not the maze generator's internal RNG.

### A2C Configuration

| Parameter | Value |
|---|---|
| Observation | 116D: frame-stacked local view (29D x 4 frames) |
| Single frame | 5x5 grid (25) + goal direction (2) + progress (1) + revisit flag (1) |
| Action space | Discrete(4): Up, Down, Left, Right |
| Network | MLP [128, 128] |
| Learning rate | 7e-4 |
| n_steps | 256 |
| Entropy coef | 0.02 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Total timesteps | 1,000,000 |
| Max steps/episode | 400 |
| RL seed | 0 (fixed, so only maze distribution differs) |

### Reward Function

| Component | Value | Purpose |
|---|---|---|
| Goal reached | +100.0 | Task completion |
| Step penalty | -0.1 | Encourage efficiency |
| Wall collision | -0.2 | Discourage invalid moves |
| Timeout | -5.0 | Penalize failure |
| Revisit cell | -0.3 | Discourage loops |
| BFS shaping | +1.0/-1.0 | Potential-based distance reward (in reward only, NOT in observation) |

### Evaluation Protocol

Four conditions tested on 50 unseen mazes each:
1. **Intra-PRNG:** AgentPRNG on unseen PRNG mazes
2. **Cross-PRNG:** AgentPRNG on unseen QRNG mazes
3. **Intra-QRNG:** AgentQRNG on unseen QRNG mazes
4. **Cross-QRNG:** AgentQRNG on unseen PRNG mazes

### Statistical Tests

- **Normality:** Shapiro-Wilk test
- **Group comparison:** Mann-Whitney U (non-normal data) or Welch t-test
- **Effect size:** Rank-biserial correlation (r)
- **Multivariate:** MANOVA with Wilk's Lambda (on maze structural metrics)
- **Significance level:** alpha = 0.05

## Project Structure

```
research-paper/
├── config/
│   └── settings.py              # All constants, hyperparameters, paths
├── seeds/
│   ├── generate_prng_seeds.py   # LCG seed generator
│   ├── fetch_qrng_seeds.py      # ANU Quantum API client
│   └── validate_seeds.py        # Shannon entropy + autocorrelation
├── maze/
│   ├── generator.py             # Recursive backtracking maze generation
│   ├── graph_utils.py           # Cell graph + corridor graph (networkx)
│   ├── metrics.py               # Path length, tortuosity, dead-ends, junctions
│   └── era.py                   # Expressive Range Analysis + convex hull
├── env/
│   └── maze_env.py              # Gymnasium env (frame-stacked local view)
├── training/
│   ├── train.py                 # A2C training script
│   └── callbacks.py             # Episode logging + checkpoints
├── evaluation/
│   └── evaluate.py              # Intra/cross-domain generalization tests
├── analysis/
│   ├── statistical_tests.py     # Shapiro-Wilk, Mann-Whitney, MANOVA
│   └── plots.py                 # Boxplots, training curves, eval summary
├── scripts/
│   ├── run_pipeline.py          # End-to-end orchestration (8 steps)
│   └── generate_gifs.py         # Maze navigation GIF visualizations
├── data/
│   ├── seeds/                   # PRNG and QRNG seed CSVs
│   ├── mazes/                   # Generated maze arrays (.npz)
│   ├── metrics/                 # All results, plots, GIFs, CSVs
│   └── models/                  # Trained A2C model checkpoints
├── context/
│   └── thesis.md                # Full thesis manuscript
└── requirements.txt
```

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Full Pipeline

```bash
# If QRNG seeds are already saved:
python3 -m scripts.run_pipeline --skip-qrng-fetch

# If you need to fetch QRNG seeds (requires internet):
python3 -m seeds.fetch_qrng_seeds
python3 -m scripts.run_pipeline --skip-qrng-fetch
```

### Step by Step

```bash
python3 -m seeds.generate_prng_seeds          # 1. Generate LCG seeds
python3 -m seeds.fetch_qrng_seeds             # 2. Fetch QRNG seeds (needs internet)
python3 -m seeds.validate_seeds               # 3. Seed quality validation
python3 -m maze.generator                     # 4. Generate all mazes
python3 -m training.train --seed-type prng    # 5. Train A2C on PRNG mazes (~5 min)
python3 -m training.train --seed-type qrng    # 6. Train A2C on QRNG mazes (~5 min)
python3 -m evaluation.evaluate                # 7. Evaluate all 4 conditions
python3 -m scripts.generate_gifs              # 8. Generate navigation GIFs
```

The pipeline supports resuming: `python3 -m scripts.run_pipeline --skip-qrng-fetch --start-from 5`

### Generated Outputs

All results are saved to `data/metrics/`:

| File | Description |
|---|---|
| `training_curves.png` | Learning curves: reward + success rate over episodes |
| `evaluation_summary.png` | Bar charts comparing all 4 evaluation conditions |
| `era_analysis.png` | Expressive Range Analysis (linearity vs leniency) |
| `seed_validation.png` | Shannon entropy + autocorrelation comparison |
| `maze_samples.png` | Visual grid of PRNG vs QRNG mazes |
| `maze_statistical_tests.csv` | All maze structure statistical test results |
| `rl_statistical_tests.csv` | All RL performance statistical test results |
| `gif_comparison_maze*.gif` | Side-by-side agent navigation animations |
| `gif_*_on_*_maze*.gif` | Individual agent navigation GIFs |

## Key Design Decisions

### Why LCG instead of Mersenne Twister?

The original thesis used Python's `random` module (Mersenne Twister, period 2^19937-1) as the PRNG baseline. Initial experiments showed **no significant difference** vs QRNG because MT is statistically too strong for 7x7 maze generation. The LCG (mod 2^32) has real sequential correlation that propagates into maze structure, creating a genuine entropy contrast.

### Why 7x7 instead of 20x20?

A2C with an MLP policy cannot learn 20x20 maze navigation from partial observations within feasible training time. The 7x7 size provides the optimal difficulty: agents achieve 60-90% success (not trivially easy, not impossibly hard), leaving room for PRNG vs QRNG training differences to manifest.

### Why Frame Stacking?

The 5x5 local grid view alone causes perceptual aliasing (different locations look identical). Stacking the last 4 frames gives the MLP temporal context to infer movement direction and detect backtracking, without requiring a recurrent (LSTM) policy.

### Why BFS Reward Shaping but NOT in Observation?

BFS distances in the **observation** trivialize the task (agent just follows the gradient, 100% on any maze, no PRNG/QRNG difference). BFS distances in the **reward** guide learning without giving the agent a "cheat code" — it must still learn navigation patterns from the local view, where training maze diversity matters.

## Guide for Groupmates: Adapting to PPO and DQN

This codebase is designed so that **only `training/train.py` and `config/settings.py` need to change** to swap algorithms. Everything else (environment, maze generation, seeds, evaluation, statistics) stays identical — ensuring a fair comparison across all three DRL architectures.

### Step 1: Copy the Repository

Copy the entire repository. The shared components are:
- `env/maze_env.py` — **DO NOT MODIFY** (must be identical across all three algorithms)
- `maze/` — maze generation and metrics (identical)
- `seeds/` — seed generation (identical, reuse the same seed CSVs)
- `evaluation/evaluate.py` — evaluation protocol (identical)
- `analysis/` — statistical tests and plots (identical)

### Step 2: Update `config/settings.py`

Replace the `A2C_CONFIG` block with the appropriate algorithm config:

**For PPO (Aldrich):**
```python
# ── PPO Hyperparameters ──────────────────────────────────────────────────
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [256, 256]},
}
TOTAL_TIMESTEPS = 1_000_000  # may need more, start here
```

**For DQN (Kevin):**
```python
# ── DQN Hyperparameters ──────────────────────────────────────────────────
DQN_CONFIG = {
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "gamma": 0.99,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "target_update_interval": 1000,
    "policy_kwargs": {"net_arch": [256, 256]},
}
TOTAL_TIMESTEPS = 1_000_000  # DQN is more sample-efficient, may need less
```

### Step 3: Update `training/train.py`

Replace the A2C import and initialization:

**For PPO:**
```python
from stable_baselines3 import PPO
# ...
model = PPO(
    policy="MlpPolicy",
    env=env,
    seed=RL_SEED,
    verbose=0,
    **PPO_CONFIG,
)
```

**For DQN:**
```python
from stable_baselines3 import DQN
# ...
model = DQN(
    policy="MlpPolicy",
    env=env,
    seed=RL_SEED,
    verbose=0,
    **DQN_CONFIG,
)
```

Also update model loading in `evaluation/evaluate.py`:
```python
from stable_baselines3 import PPO  # or DQN
prng_model = PPO.load(...)  # instead of A2C.load(...)
```

### Step 4: Run the Pipeline

Same commands, everything else is automated:
```bash
python3 -m scripts.run_pipeline --skip-qrng-fetch
```

### What MUST Stay Identical Across All Three Algorithms

These ensure a fair cross-algorithm comparison:

| Component | Why |
|---|---|
| `env/maze_env.py` | Same observation space, rewards, and dynamics |
| Maze seeds (CSVs) | Same PRNG and QRNG seed sets |
| Maze generation | Same recursive backtracking algorithm |
| Train/test split | Same 50/50 split, same indices |
| `RL_SEED = 0` | Fixed random seed for reproducibility |
| Evaluation protocol | Same 4 conditions, same test mazes |
| Statistical tests | Same alpha, same methods |

### Expected Differences Between Algorithms

| Aspect | A2C | PPO | DQN |
|---|---|---|---|
| Sample efficiency | Lowest | Medium | Highest (replay buffer) |
| Training stability | Can oscillate | Most stable (clipping) | Stable (target network) |
| Convergence speed | Slowest | Medium | Fastest for discrete actions |
| Recommended timesteps | 1M | 1M | 500K-1M |

The thesis does NOT compare algorithms against each other. Each algorithm independently tests whether QRNG-trained agents outperform PRNG-trained agents. Consistent results across all three algorithms would strongly support the hypothesis.

## Team

| Member | Algorithm | Role |
|---|---|---|
| Mark Angelo R. Baricante | **A2C** | This repository |
| Aldrich Ryan V. Antony | PPO | Separate repository |
| Kevin Hans Aurick S. Mirabel | DQN | Separate repository |

**Supervisor:** John Richard M. Esguerra, MSCS

## Tech Stack

- Python 3.14
- Stable-Baselines3 (A2C)
- Gymnasium (custom environment)
- NumPy, NetworkX, SciPy, Statsmodels
- Matplotlib, Pillow (visualization)
