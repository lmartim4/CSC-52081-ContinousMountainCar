"""Wrapper for symbolic (RAM-based) grid observation.

Builds a compact 13x16 grid directly from the NES RAM, encoding tiles,
enemies, and Mario's position — no pixel processing needed.

Includes the same reward shaping as the pixel pipeline (vietnh1009)
so that pixel vs RAM comparisons are fair.

Delegates grid construction to the external submodule:
  external/yumouwei-smb/smb_utils.py  (smb_grid class)

References:
  - RAM map: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
  - Tile grid: yumouwei/super-mario-bros-reinforcement-learning (smb_utils.py)
  - Reward shaping: vietnh1009/Super-mario-bros-PPO-pytorch
"""

import gym
import numpy as np
from gym import spaces

from ..utils.smb_utils import smb_grid

# Grid encoding values (same convention as smb_grid, plus powerup)
EMPTY = 0
SOLID = 1
ENEMY = -1
MARIO = 2
POWERUP = 3

# Powerup RAM addresses
_POWERUP_DRAWN = 0x0014
_POWERUP_X_SCREEN = 0x008C
_POWERUP_Y_SCREEN = 0x00D4

# Visible grid dimensions (what the screen shows)
VISIBLE_COLS = 16
VISIBLE_ROWS = 13


class CustomRewardRAM(gym.Wrapper):
    """Custom reward shaping (same as pixel pipeline for fair comparison).

    - Adds score-based reward: (score_delta) / 40
    - Flag reached: +50
    - Death / timeout: -50
    - All rewards scaled by /10
    """

    def __init__(self, env):
        super().__init__(env)
        self.curr_score = 0
        self.current_x = 40

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False

        # Score-based reward
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]

        # Flag / death bonus
        if done:
            if info.get("flag_get", False):
                reward += 50
            else:
                reward -= 50

        self.current_x = info.get("x_pos", self.current_x)
        return obs, reward / 10.0, done, truncated, info

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        self.curr_score = 0
        self.current_x = 40
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return obs, {}


class RAMGridObservation(gym.ObservationWrapper):
    """Convert the pixel observation into a 13x16 symbolic grid read from RAM.

    Delegates to smb_grid from external/yumouwei-smb/smb_utils.py.

    The grid encodes:
       0 = empty / sky
       1 = solid tile (ground, brick, pipe, block, etc.)
      -1 = enemy
       2 = Mario

    The observation is a float32 array of shape (13, 16).
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-1, high=3,
            shape=(VISIBLE_ROWS, VISIBLE_COLS),
            dtype=np.float32,
        )

    def observation(self, obs):
        """Ignore the pixel obs; build grid from RAM via smb_grid."""
        grid = smb_grid(self.env).rendered_screen
        self._fill_powerup(grid, self.unwrapped.ram)
        return grid.astype(np.float32)

    @staticmethod
    def _fill_powerup(grid, ram):
        """Place a powerup (mushroom / flower / star) on the grid."""
        if ram[_POWERUP_DRAWN] != 1:
            return
        px = int(ram[_POWERUP_X_SCREEN])
        py = int(ram[_POWERUP_Y_SCREEN])
        col = (px + 8) // 16
        row = (py + 8 - 32) // 16
        if 0 <= row < VISIBLE_ROWS and 0 <= col < VISIBLE_COLS:
            grid[row, col] = POWERUP


class FlattenGrid(gym.ObservationWrapper):
    """Flatten the 2-D grid into a 1-D vector for MLP policies."""

    def __init__(self, env):
        super().__init__(env)
        flat_size = int(np.prod(self.observation_space.shape))
        self.observation_space = spaces.Box(
            low=-1, high=3,
            shape=(flat_size,),
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.flatten()


class FrameStackGrid(gym.Wrapper):
    """Stack the last *n_stack* grid frames along a new last axis.

    Resulting shape: (13, 16, n_stack).
    This gives the agent temporal information (velocity, direction).
    """

    def __init__(self, env, n_stack=4, n_skip=1):
        super().__init__(env)
        self.n_stack = n_stack
        self.n_skip = n_skip
        base_shape = env.observation_space.shape  # (13, 16)
        self.observation_space = spaces.Box(
            low=-1, high=3,
            shape=(*base_shape, n_stack),
            dtype=np.float32,
        )
        self._frames = []

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})
        self._frames = [obs] * (self.n_stack * self.n_skip)
        return self._get_stacked(), info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, info = result
            truncated = False
        self._frames.append(obs)
        if len(self._frames) > self.n_stack * self.n_skip:
            self._frames.pop(0)
        return self._get_stacked(), reward, terminated, truncated, info

    def _get_stacked(self):
        # Pick every n_skip-th frame from the buffer
        indices = [
            len(self._frames) - 1 - i * self.n_skip
            for i in range(self.n_stack)
        ]
        indices = sorted(indices)
        selected = [self._frames[max(0, i)] for i in indices]
        return np.stack(selected, axis=-1).astype(np.float32)


class SkipFrame(gym.Wrapper):
    """Repeat the same action for *skip* frames and sum the rewards."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})
        return obs, info

    def step(self, action):
        total_reward = 0.0
        terminated, truncated = False, False
        for _ in range(self.skip):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward, terminated, info = result
                truncated = False
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Helper: build a ready-to-use symbolic environment
# ---------------------------------------------------------------------------

def make_symbolic_env(
    env_id="SuperMarioBros-1-1-v3",
    skip=4,
    n_stack=4,
    n_skip=1,
    flatten=False,
):
    """Create a symbolic-observation Super Mario Bros environment.

    Pipeline: JoypadSpace -> CustomRewardRAM -> SkipFrame -> RAMGridObservation
              -> FrameStackGrid -> FlattenGrid
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomRewardRAM(env)
    env = SkipFrame(env, skip=skip)
    env = RAMGridObservation(env)

    if n_stack > 1:
        env = FrameStackGrid(env, n_stack=n_stack, n_skip=n_skip)

    if flatten:
        env = FlattenGrid(env)

    return env


def make_symbolic_vec_env(
    env_id="SuperMarioBros-1-1-v3",
    skip=4,
    n_stack=4,
    n_skip=1,
    flatten=False,
    num_envs=8,
):
    """Create parallel symbolic-observation environments using SubprocVecEnv."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def _make_env(env_id, skip, n_stack, n_skip, flatten):
        def _init():
            return make_symbolic_env(
                env_id=env_id, skip=skip,
                n_stack=n_stack, n_skip=n_skip, flatten=flatten,
            )
        return _init

    return SubprocVecEnv([
        _make_env(env_id, skip, n_stack, n_skip, flatten)
        for _ in range(num_envs)
    ])
