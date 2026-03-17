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
from collections import deque
from gym import spaces

from ..utils.smb_utils import smb_grid

# Grid encoding values (same convention as smb_grid, plus powerup and fire)
EMPTY   =  0
SOLID   =  1
ENEMY   = -1
MARIO   =  2
POWERUP =  3
FIRE    =  4   # fire-snake bead (OAM tile 0x65)

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
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, info = result
            truncated = False

        # Score-based reward
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]

        # Flag / death / timeout bonus
        if terminated or truncated:
            if info.get("flag_get", False):
                reward += 50
            else:
                reward -= 50

        self.current_x = info.get("x_pos", self.current_x)
        return obs, reward / 10.0, terminated, truncated, info

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
            low=-1, high=4,
            shape=(VISIBLE_ROWS, VISIBLE_COLS),
            dtype=np.float32,
        )
        self._grid = smb_grid(env)  # created once; re-initialised each step

    def observation(self, obs):
        """Ignore the pixel obs; build grid from RAM via smb_grid."""
        self._grid.__init__(self.env)
        grid = self._grid.rendered_screen.copy()
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
    """Flatten the 2-D grid into a 1-D vector for MLP policies.

    Appends Mario's normalised power state (RAM 0x0756 / 2) as an extra
    element, growing the observation from (832,) to (833,).
    """

    # RAM address for Mario's power state: 0=small, 1=super, 2=fire
    _POWER_STATE_ADDR = 0x0756

    def __init__(self, env):
        super().__init__(env)
        flat_size = int(np.prod(self.observation_space.shape)) + 1
        low = np.full((flat_size,), -1, dtype=np.float32)
        high = np.full((flat_size,), 4, dtype=np.float32)
        low[-1] = 0.0
        high[-1] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        power = self.unwrapped.ram[self._POWER_STATE_ADDR] / 2.0
        return np.append(obs.flatten(), np.float32(power))


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
            low=-1, high=4,
            shape=(*base_shape, n_stack),
            dtype=np.float32,
        )
        buf_size = self.n_stack * self.n_skip
        self._frames = deque(maxlen=buf_size)

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})
        self._frames.extend([obs] * self._frames.maxlen)
        return self._get_stacked(), info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, info = result
            truncated = False
        self._frames.append(obs)
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


class RandomStartWrapper(gym.Wrapper):
    """Advance Mario by a random number of steps at the start of each episode.

    Samples n_steps ~ Uniform[0, max_start_steps] and steps the env with
    forward-biased random actions (right / right+A / right+B / right+A+B)
    before returning the starting observation.  If Mario dies during warmup,
    a fresh reset state is returned instead.

    Placing this after the full observation pipeline (SkipFrame, RAMGrid,
    FrameStack, Flatten) means the returned obs is always in the expected
    format.

    This prevents the agent from memorising the fixed level opening and
    forces it to learn reusable features (jumping, enemy avoidance) that
    generalise to any position.
    """

    # SIMPLE_MOVEMENT indices that move Mario rightward.
    # Excludes NOOP (0), stationary jump (5), and left (6) so warmup
    # actually advances Mario into the level.
    _FORWARD_ACTIONS = [1, 2, 3, 4]  # right, right+A, right+B, right+A+B

    def __init__(self, env, max_start_steps: int = 100):
        super().__init__(env)
        self.max_start_steps = max_start_steps

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})

        if self.max_start_steps <= 0:
            return obs, info

        n_steps = np.random.randint(0, self.max_start_steps + 1)
        for _ in range(n_steps):
            action = int(np.random.choice(self._FORWARD_ACTIONS))
            result = self.env.step(action)
            if len(result) == 5:
                obs, _, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, _, done, info = result
            if done:
                # Mario died during warmup — return a clean reset
                result = self.env.reset(**kwargs)
                obs, info = result if isinstance(result, tuple) else (result, {})
                break

        return obs, info

    def set_max_start_steps(self, value: int):
        self.max_start_steps = int(value)

    def step(self, action):
        return self.env.step(action)


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
    monitor=False,
    random_start_steps=0,
):
    """Create a symbolic-observation Super Mario Bros environment.

    Pipeline: JoypadSpace -> CustomRewardRAM -> SkipFrame -> RAMGridObservation
              -> FrameStackGrid -> FlattenGrid -> RandomStartWrapper -> Monitor

    Args:
        random_start_steps: If > 0, a RandomStartWrapper is added that
            advances Mario by Uniform[0, random_start_steps] steps at the
            start of each episode, producing diverse starting positions.
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

    if random_start_steps > 0:
        env = RandomStartWrapper(env, max_start_steps=random_start_steps)

    if monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)

    return env


def make_symbolic_vec_env(
    env_id="SuperMarioBros-1-1-v3",
    skip=4,
    n_stack=4,
    n_skip=1,
    flatten=False,
    num_envs=8,
    random_start_steps=0,
):
    """Create parallel symbolic-observation environments using SubprocVecEnv."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def _make_env(env_id, skip, n_stack, n_skip, flatten, random_start_steps):
        def _init():
            return make_symbolic_env(
                env_id=env_id, skip=skip,
                n_stack=n_stack, n_skip=n_skip, flatten=flatten,
                random_start_steps=random_start_steps,
            )
        return _init

    return SubprocVecEnv([
        _make_env(env_id, skip, n_stack, n_skip, flatten, random_start_steps)
        for _ in range(num_envs)
    ])


def make_symbolic_multitask_vec_env(
    env_ids=("SuperMarioBros-1-1-v3", "SuperMarioBros-1-2-v3"),
    skip=4,
    n_stack=4,
    n_skip=1,
    flatten=False,
    envs_per_level=4,
    random_start_steps=0,
):
    """Create parallel envs across multiple levels for multi-task training.

    Args:
        envs_per_level: int or list of ints. If int, same count for all levels.
            If list, specifies count per level (must match len(env_ids)).
        random_start_steps: If > 0, each env uses RandomStartWrapper to spawn
            Mario at a random position up to this many steps into the level.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if isinstance(envs_per_level, int):
        counts = [envs_per_level] * len(env_ids)
    else:
        counts = list(envs_per_level)

    def _make_env(env_id, skip, n_stack, n_skip, flatten, random_start_steps):
        def _init():
            return make_symbolic_env(
                env_id=env_id, skip=skip,
                n_stack=n_stack, n_skip=n_skip, flatten=flatten,
                random_start_steps=random_start_steps,
            )
        return _init

    env_fns = []
    for eid, count in zip(env_ids, counts):
        for _ in range(count):
            env_fns.append(_make_env(eid, skip, n_stack, n_skip, flatten, random_start_steps))

    return SubprocVecEnv(env_fns)
