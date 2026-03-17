"""A2C training script for maze navigation.

Usage:
    python -m training.train --seed-type prng
    python -m training.train --seed-type qrng
"""

import argparse

import numpy as np
from stable_baselines3 import A2C

from config.settings import (
    A2C_CONFIG,
    CHECKPOINT_FREQ,
    MODELS_DIR,
    METRICS_DIR,
    NUM_TRAIN,
    RL_SEED,
    TOTAL_TIMESTEPS,
)
from env.maze_env import MazeEnv
from maze.generator import load_mazes
from training.callbacks import MazeTrainingCallback


def train_a2c(seed_type: str) -> None:
    """Train an A2C agent on mazes of the given seed type."""
    print(f"\n{'='*60}")
    print(f"Training A2C on {seed_type.upper()} mazes")
    print(f"{'='*60}")

    # Load mazes and split
    all_mazes = load_mazes(seed_type)
    train_mazes = [all_mazes[i] for i in range(NUM_TRAIN)]
    print(f"  Training on {len(train_mazes)} mazes, total timesteps: {TOTAL_TIMESTEPS:,}")

    # Create environment
    env = MazeEnv(mazes=train_mazes)

    # Set up logging
    log_path = METRICS_DIR / f"a2c_{seed_type}_training_log.csv"
    checkpoint_dir = MODELS_DIR / f"a2c_{seed_type}" / "checkpoints"

    callback = MazeTrainingCallback(
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=CHECKPOINT_FREQ,
    )

    # Initialize A2C
    model = A2C(
        policy="MlpPolicy",
        env=env,
        seed=RL_SEED,
        verbose=0,
        **A2C_CONFIG,
    )

    print(f"  Model architecture: {A2C_CONFIG['policy_kwargs']['net_arch']}")
    print(f"  Learning rate: {A2C_CONFIG['learning_rate']}")
    print(f"  Training...")

    # Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # Save final model
    final_path = MODELS_DIR / f"a2c_{seed_type}" / "final_model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    print(f"\n  Final model saved: {final_path}")
    print(f"  Training log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train A2C on maze environments")
    parser.add_argument(
        "--seed-type",
        choices=["prng", "qrng"],
        required=True,
        help="Which seed group to train on",
    )
    args = parser.parse_args()
    train_a2c(args.seed_type)


if __name__ == "__main__":
    main()
