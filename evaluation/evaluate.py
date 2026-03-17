"""Evaluate trained A2C agents on unseen mazes.

Runs intra-domain and cross-domain generalization tests:
  - AgentPRNG on unseen PRNG mazes (intra)
  - AgentPRNG on unseen QRNG mazes (cross)
  - AgentQRNG on unseen QRNG mazes (intra)
  - AgentQRNG on unseen PRNG mazes (cross)
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import A2C

from config.settings import MAX_STEPS, METRICS_DIR, MODELS_DIR, NUM_TRAIN
from env.maze_env import MazeEnv
from maze.generator import load_mazes


def evaluate_agent(
    model: A2C,
    test_mazes: list[np.ndarray],
    deterministic: bool = True,
) -> pd.DataFrame:
    """Run the agent on each test maze and record results.

    Returns DataFrame with columns: maze_index, success, reward, steps.
    """
    records = []

    for i, maze in enumerate(test_mazes):
        # Create fresh env per maze so BFS distance maps are correct
        env = MazeEnv(mazes=[maze])
        obs, _ = env.reset()

        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        records.append({
            "maze_index": i,
            "success": int(info.get("success", False)),
            "reward": round(total_reward, 4),
            "steps": steps,
        })

    return pd.DataFrame(records)


def run_all_evaluations() -> dict[str, pd.DataFrame]:
    """Run all 4 evaluation conditions and save results."""
    results = {}

    # Load test mazes (indices NUM_TRAIN onward)
    prng_mazes = load_mazes("prng")
    qrng_mazes = load_mazes("qrng")
    prng_test = [prng_mazes[i] for i in range(NUM_TRAIN, len(prng_mazes))]
    qrng_test = [qrng_mazes[i] for i in range(NUM_TRAIN, len(qrng_mazes))]

    # Load models
    prng_model = A2C.load(str(MODELS_DIR / "a2c_prng" / "final_model"))
    qrng_model = A2C.load(str(MODELS_DIR / "a2c_qrng" / "final_model"))

    conditions = [
        ("a2c_prng_on_prng", prng_model, prng_test, "intra"),
        ("a2c_prng_on_qrng", prng_model, qrng_test, "cross"),
        ("a2c_qrng_on_qrng", qrng_model, qrng_test, "intra"),
        ("a2c_qrng_on_prng", qrng_model, prng_test, "cross"),
    ]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model, test_mazes, domain_type in conditions:
        print(f"\nEvaluating: {name} ({domain_type}-domain)")
        df = evaluate_agent(model, test_mazes)
        out_path = METRICS_DIR / f"eval_{name}.csv"
        df.to_csv(out_path, index=False)

        sr = df["success"].mean() * 100
        mean_reward = df["reward"].mean()
        mean_steps = df["steps"].mean()
        print(f"  Success Rate: {sr:.1f}%")
        print(f"  Mean Reward:  {mean_reward:.2f}")
        print(f"  Mean Steps:   {mean_steps:.1f}")

        results[name] = df

    # Generalization gap
    print(f"\n{'='*60}")
    print("Generalization Gap (SR_intra - SR_cross):")
    sr_prng_intra = results["a2c_prng_on_prng"]["success"].mean() * 100
    sr_prng_cross = results["a2c_prng_on_qrng"]["success"].mean() * 100
    sr_qrng_intra = results["a2c_qrng_on_qrng"]["success"].mean() * 100
    sr_qrng_cross = results["a2c_qrng_on_prng"]["success"].mean() * 100

    print(f"  AgentPRNG: {sr_prng_intra:.1f}% - {sr_prng_cross:.1f}% = {sr_prng_intra - sr_prng_cross:.1f}%")
    print(f"  AgentQRNG: {sr_qrng_intra:.1f}% - {sr_qrng_cross:.1f}% = {sr_qrng_intra - sr_qrng_cross:.1f}%")

    return results


def main():
    run_all_evaluations()


if __name__ == "__main__":
    main()
