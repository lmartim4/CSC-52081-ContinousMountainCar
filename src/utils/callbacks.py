"""Custom callbacks for training."""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointAndLogCallback(BaseCallback):
    """Save model checkpoints and log episode metrics during training."""

    def __init__(self, save_path, save_freq=50_000, log_freq=1_000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_flags = []  # x_pos at end of episode

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Collect episode info from the Monitor wrapper
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if "flag_get" in info:
                self.episode_flags.append(info["flag_get"])

        # Checkpoint
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(path)
            if self.verbose:
                print(f"[Checkpoint] Saved model at step {self.n_calls} → {path}")

        # Logging
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            mean_r = np.mean(self.episode_rewards[-100:])
            mean_l = np.mean(self.episode_lengths[-100:])
            self.logger.record("rollout/mean_reward_100", mean_r)
            self.logger.record("rollout/mean_length_100", mean_l)
            if len(self.episode_flags) > 0:
                flag_rate = np.mean(self.episode_flags[-100:])
                self.logger.record("rollout/flag_rate_100", flag_rate)

        return True
