"""Custom callbacks for training."""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointAndLogCallback(BaseCallback):
    """Save model checkpoints and log episode metrics during training.

    Tracks episode rewards/lengths/flags directly from env info dicts,
    without requiring a Monitor wrapper.
    """

    def __init__(self, save_path, save_freq=50_000, log_freq=1_000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_flags = []
        # Per-env accumulators
        self._env_rewards = None
        self._env_lengths = None
        self._env_flags = None

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)
        n_envs = self.training_env.num_envs
        self._env_rewards = np.zeros(n_envs, dtype=np.float64)
        self._env_lengths = np.zeros(n_envs, dtype=np.int64)
        self._env_flags = np.zeros(n_envs, dtype=bool)

    def _on_step(self):
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, (r, d, info) in enumerate(zip(rewards, dones, infos)):
            self._env_rewards[i] += r
            self._env_lengths[i] += 1
            if info.get("flag_get", False):
                self._env_flags[i] = True

            if d:
                self.episode_rewards.append(float(self._env_rewards[i]))
                self.episode_lengths.append(int(self._env_lengths[i]))
                self.episode_flags.append(bool(self._env_flags[i]))
                self._env_rewards[i] = 0.0
                self._env_lengths[i] = 0
                self._env_flags[i] = False

        # Checkpoint
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(path)
            if self.verbose:
                print(f"[Checkpoint] Saved model at step {self.n_calls} → {path}")

        # Logging to TensorBoard
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            last_100_r = self.episode_rewards[-100:]
            last_100_l = self.episode_lengths[-100:]
            last_100_f = self.episode_flags[-100:]
            self.logger.record("rollout/ep_rew_mean", np.mean(last_100_r))
            self.logger.record("rollout/ep_len_mean", np.mean(last_100_l))
            self.logger.record("rollout/flag_rate_100", np.mean(last_100_f))

        return True
