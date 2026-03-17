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


class CurriculumCallback(BaseCallback):
    """Linearly increase random_start_steps as training progresses."""

    def __init__(self, start_steps=0, end_steps=100, total_timesteps=4_000_000):
        super().__init__()
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            fraction = min(self.n_calls / self.total_timesteps, 1.0)
            current = int(self.start_steps + fraction * (self.end_steps - self.start_steps))
            self.training_env.env_method('set_max_start_steps', current)
        return True


class PerLevelEvalCallback(BaseCallback):
    """Periodically evaluate the model on each level and log separately."""

    def __init__(self, levels, eval_freq=100_000, n_eval_episodes=5, skip=4, n_stack=4):
        super().__init__()
        self.levels = levels
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.skip = skip
        self.n_stack = n_stack

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        from src.wrappers import make_symbolic_env

        for level in self.levels:
            env = make_symbolic_env(
                env_id=level, skip=self.skip, n_stack=self.n_stack,
                flatten=True, random_start_steps=0,
            )
            rewards, flags, x_positions = [], [], []

            for _ in range(self.n_eval_episodes):
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result
                done, total_reward, flag, max_x = False, 0.0, False, 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    result = env.step(int(action))
                    if len(result) == 5:
                        obs, reward, terminated, truncated, info = result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = result
                    total_reward += float(reward)
                    max_x = max(max_x, info.get('x_pos', 0))
                    if info.get('flag_get', False):
                        flag = True

                rewards.append(total_reward)
                flags.append(flag)
                x_positions.append(max_x)

            env.close()

            tag = level.split('-')[1] + '-' + level.split('-')[2]
            self.logger.record(f'eval/{tag}_mean_reward', np.mean(rewards))
            self.logger.record(f'eval/{tag}_flag_rate',   np.mean(flags))
            self.logger.record(f'eval/{tag}_mean_x_pos',  np.mean(x_positions))

        self.logger.dump(self.num_timesteps)
        return True
