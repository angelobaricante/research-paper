"""Training callbacks for logging and checkpointing."""

import csv
from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class MazeTrainingCallback(BaseCallback):
    """Logs per-episode metrics and saves model checkpoints.

    Prints a rolling success rate every N episodes so you can
    track learning progress during training.
    """

    def __init__(
        self,
        log_path: Path,
        checkpoint_dir: Path,
        checkpoint_freq: int = 10_000,
        print_freq: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.print_freq = print_freq

        self._episode_rewards = 0.0
        self._episode_steps = 0
        self._episode_count = 0
        self._total_successes = 0
        self._recent_successes = deque(maxlen=100)  # rolling window
        self._recent_rewards = deque(maxlen=100)
        self._csv_file = None
        self._csv_writer = None

    def _on_training_start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._csv_file = open(self.log_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "episode", "timestep", "reward", "steps", "success",
        ])

    def _on_step(self) -> bool:
        # Accumulate reward for current episode
        self._episode_rewards += self.locals["rewards"][0]
        self._episode_steps += 1

        # Check for episode end
        dones = self.locals.get("dones", self.locals.get("done", [False]))
        infos = self.locals.get("infos", [{}])

        if dones[0]:
            self._episode_count += 1
            success = infos[0].get("success", False)
            self._total_successes += int(success)
            self._recent_successes.append(int(success))
            self._recent_rewards.append(self._episode_rewards)

            self._csv_writer.writerow([
                self._episode_count,
                self.num_timesteps,
                round(self._episode_rewards, 4),
                self._episode_steps,
                int(success),
            ])
            self._csv_file.flush()

            if self.verbose >= 1 and self._episode_count % self.print_freq == 0:
                sr = sum(self._recent_successes) / len(self._recent_successes) * 100
                avg_r = sum(self._recent_rewards) / len(self._recent_rewards)
                total_sr = self._total_successes / self._episode_count * 100
                print(
                    f"  Ep {self._episode_count:>5d} | "
                    f"Step {self.num_timesteps:>8,d} | "
                    f"SR(100) {sr:5.1f}% | "
                    f"SR(all) {total_sr:5.1f}% | "
                    f"Reward(100) {avg_r:>7.1f} | "
                    f"Steps {self._episode_steps:>4d}"
                )

            self._episode_rewards = 0.0
            self._episode_steps = 0

        # Checkpoint
        if self.num_timesteps % self.checkpoint_freq == 0:
            path = self.checkpoint_dir / f"checkpoint_{self.num_timesteps}"
            self.model.save(str(path))

        return True

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()
        total_sr = self._total_successes / max(self._episode_count, 1) * 100
        print(f"\n  Training complete: {self._episode_count} episodes, "
              f"{self._total_successes} successes ({total_sr:.1f}%)")
