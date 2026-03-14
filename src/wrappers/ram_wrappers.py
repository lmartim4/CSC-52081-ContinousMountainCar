"""Wrapper for symbolic (RAM-based) grid observation.

Inspired by yumouwei's smb_utils.py — extracts a tile grid from the
NES RAM to build a compact symbolic representation of the level.
"""

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
from gym import spaces

from src.config import GRID_SHAPE, ENV_ID, SIMPLE_MOVEMENT as USE_SIMPLE


# --- RAM addresses for Super Mario Bros ---
# Reference: https://datacrystal.romhacking.net/wiki/Super_Mario_Bros.:RAM_map
PLAYER_X_ADDR = 0x006D
PLAYER_Y_ADDR = 0x00CE
PLAYER_STATE_ADDR = 0x000E
CURRENT_PAGE_ADDR = 0x006D  # player x position within page
SCREEN_X_POS_ADDR = 0x03AD
ENEMY_DRAWN_ADDR = 0x000F  # 5 enemy slots starting here
ENEMY_X_ADDR = 0x0087      # 5 enemy slots
ENEMY_Y_ADDR = 0x00CF      # 5 enemy slots


class SymbolicGridObservation(gym.ObservationWrapper):
    """Convert pixel observation + RAM info into a symbolic grid.

    The grid encodes:
      0 = empty / sky
      1 = solid block / ground
      2 = enemy
      3 = Mario
      4 = item / power-up
      5 = pipe
    """

    def __init__(self, env, grid_shape=GRID_SHAPE):
        super().__init__(env)
        self.grid_rows, self.grid_cols = grid_shape
        self.observation_space = spaces.Box(
            low=0, high=5,
            shape=(self.grid_rows, self.grid_cols),
            dtype=np.float32,
        )

    def observation(self, obs):
        ram = self.unwrapped.ram if hasattr(self.unwrapped, 'ram') else None
        grid = self._build_grid(obs, ram)
        return grid.astype(np.float32)

    def _build_grid(self, obs, ram):
        """Build symbolic grid from the pixel frame.

        Uses a simple tile-based approach: divide the screen into a grid
        and classify each tile by its dominant colour content.
        """
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        if obs is None:
            return grid

        h, w = obs.shape[:2]
        tile_h = h // self.grid_rows
        tile_w = w // self.grid_cols

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                tile = obs[r * tile_h:(r + 1) * tile_h,
                           c * tile_w:(c + 1) * tile_w]
                grid[r, c] = self._classify_tile(tile)

        return grid

    @staticmethod
    def _classify_tile(tile):
        """Classify a tile based on colour statistics.

        This is a heuristic — can be improved with RAM-based classification.
        """
        if tile.size == 0:
            return 0

        mean = tile.mean(axis=(0, 1)) if tile.ndim == 3 else tile.mean()

        if tile.ndim == 3 and tile.shape[2] >= 3:
            r, g, b = mean[0], mean[1], mean[2]
            # Sky (blue-ish)
            if b > 150 and r < 100 and g < 150:
                return 0
            # Ground / brick (brown-ish)
            if r > 150 and g < 120 and b < 80:
                return 1
            # Enemy (red/orange)
            if r > 180 and g < 100:
                return 2
            # Mario (red dominant but small region)
            if r > 150 and g < 80 and b < 80:
                return 3
            # Pipe (green)
            if g > 150 and r < 100 and b < 100:
                return 5
            # Item (yellow-ish)
            if r > 200 and g > 200 and b < 100:
                return 4
        else:
            val = float(mean)
            if val > 200:
                return 0  # bright = sky
            elif val > 100:
                return 1  # medium = ground
        return 0


class FlattenGrid(gym.ObservationWrapper):
    """Flatten the 2D grid into a 1D vector for MLP policies."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=self.observation_space.low.flatten(),
            high=self.observation_space.high.flatten(),
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.flatten()


def make_symbolic_env(env_id=ENV_ID, grid_shape=GRID_SHAPE, skip=4, flatten=True):
    """Create a symbolic-observation Mario environment."""
    from src.wrappers.pixel_wrappers import SkipFrame

    env = gym_super_mario_bros.make(env_id)
    actions = SIMPLE_MOVEMENT if USE_SIMPLE else COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = SkipFrame(env, skip=skip)
    env = SymbolicGridObservation(env, grid_shape=grid_shape)
    if flatten:
        env = FlattenGrid(env)
    return env
