"""Wrapper for symbolic (RAM-based) grid observation.

Builds a compact 13x16 grid directly from the NES RAM, encoding tiles,
enemies, and Mario's position — no pixel processing needed.

Delegates grid construction to the external submodule:
  external/yumouwei-smb/smb_utils.py  (smb_grid class)

For the augmented GameRamState, AugmentedRAMObservation wraps the grid
and appends a scalar feature vector, returning a Dict observation:
  {
    "grid":    float32 (13, 16)  — spatial tile/enemy/powerup layout
    "scalars": float32 (N,)      — Mario state, physics, game timer, …
  }

Scalar feature layout (batch 1 — 6 values, all normalised to [-1, 1] or [0, 1]):
  [0] x_speed        0x0057  horizontal speed, signed, /40  → [-1, 1]
  [1] y_velocity     0x009F  vertical velocity, signed, /8  → [-1, 1]
  [2] float_state    0x001D  0=ground 1=jump 2=ledge 3=pole → [0, 1]
  [3] power_state    0x0756  0=small 1=big 2+=fire,  /2    → [0, 1]
  [4] inv_timer      0x079E  invincibility frames,   /255  → [0, 1]
  [5] game_timer     0x07F8-FA  BCD hundreds/tens/ones /400 → [0, 1]

References:
  - RAM map: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
  - Tile grid: yumouwei/super-mario-bros-reinforcement-learning (smb_utils.py)
"""

import gym
import numpy as np
from gym import spaces

from ..utils.smb_utils import smb_grid

# Grid encoding values
EMPTY    = 0
SOLID    = 1   # ground, pipes, platforms (non-interactive)
ENEMY    = -1
MARIO    = 2
MUSHROOM = 3   # super mushroom (red)
FLOWER   = 4   # fire flower
STAR     = 5   # super star
BRICK    = 6   # breakable brick block
QUESTION = 7   # ? block with hidden item
ONEUP    = 8   # 1-UP mushroom (green)

# Backwards-compatible alias
POWERUP  = MUSHROOM

# Raw metatile IDs from RAM 0x500 area (discovered empirically)
# 0x51 = brick  |  0x57 = brick (powerup)  |  0xC0 = ?(coin)  |  0xC1 = ?(powerup)
# 0x54 = floor  |  0x12-0x15 = pipes  |  0x60 = hidden/invisible (excluded)
# Powerup-item type from 0x0039: 0=mushroom 1=fire flower 2=star 3=1UP
_QUESTION_IDS = frozenset({0xC0, 0xC1})           # 0xC0 = ? (coin), 0xC1 = ? (powerup)
_BRICK_IDS    = frozenset({0x51, 0x57, 0xC4})     # 0x51/0x57 = brick, 0xC4 = flat brick

# Powerup RAM addresses — the powerup uses enemy slot 5 position registers
_POWERUP_DRAWN    = 0x0014   # 0x0F + 5  non-zero when powerup is active on screen
_POWERUP_KIND     = 0x0039   # powerup type: 0=mushroom, 1=fire flower, 2=star, 3=1UP
_POWERUP_PAGE     = 0x0073   # 0x6E + 5  (high byte of level x)
_POWERUP_X_SCREEN = 0x008C   # 0x87 + 5  (low byte of level x)
_POWERUP_Y_SCREEN = 0x00D4   # 0xCF + 5

# Visible grid dimensions (what the screen shows)
VISIBLE_COLS = 16
VISIBLE_ROWS = 13

# ---------------------------------------------------------------------------
# Scalar feature RAM addresses (batch 1)
# ---------------------------------------------------------------------------
_MARIO_X_SPEED   = 0x0057  # signed byte: 0xD8…0x00…0x28  (~[-40, +40])
_MARIO_Y_VEL     = 0x009F  # signed byte: 0xFB=jump peak, 0x05=fastest fall
_MARIO_FLOAT     = 0x001D  # 0=ground, 1=jumping, 2=walking off, 3=flagpole
_MARIO_POWER     = 0x0756  # 0=small, 1=big, 2+=fire
_MARIO_INV_TIMER = 0x079E  # invincibility frames remaining (0–255)
_TIMER_HUNDREDS  = 0x07F8  # game clock BCD digits
_TIMER_TENS      = 0x07F9
_TIMER_ONES      = 0x07FA

NUM_SCALARS_BATCH1 = 6

# Metadata for each scalar feature — shared by visualisation scripts.
# Each tuple: (index, label, unit_hint, vmin, vmax, cmap_name)
SCALAR_META = [
    (0, "x_speed",    "[-1=left  +1=right]", -1.0, 1.0, "RdYlGn"),
    (1, "y_velocity", "[-1=up    +1=down]",  -1.0, 1.0, "RdYlGn_r"),
    (2, "float_state","[0=ground … 1=pole]",  0.0, 1.0, "coolwarm"),
    (3, "power_state","[0=small  … 1=fire]",  0.0, 1.0, "YlOrRd"),
    (4, "inv_timer",  "[0=none   … 1=max]",   0.0, 1.0, "Purples"),
    (5, "game_timer", "[0=done   … 1=full]",  0.0, 1.0, "Blues"),
]


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
            low=-1, high=8,
            shape=(VISIBLE_ROWS, VISIBLE_COLS),
            dtype=np.float32,
        )

    def observation(self, obs):
        """Ignore the pixel obs; build grid from RAM via smb_grid."""
        grid = smb_grid(self.env).rendered_screen
        self._classify_tiles(grid)
        self._fill_powerup(grid, self.unwrapped.ram)
        return grid.astype(np.float32)

    # Hidden/invisible block IDs — not shown to the agent.
    # 0x60 = invisible block (no pixel render, but solid in RAM).
    # 0xC2–0xCF = other invisible block variants, except 0xC4 (flat brick, visible).
    _HIDDEN_IDS = frozenset({0x60, *(v for v in range(0xC2, 0xD0) if v != 0xC4)})

    @staticmethod
    def _classify_tiles(grid):
        """Map raw metatile IDs to symbolic tile constants in-place."""
        for r in range(VISIBLE_ROWS):
            for c in range(VISIBLE_COLS):
                v = int(grid[r, c])
                if v == 0 or v in RAMGridObservation._HIDDEN_IDS:
                    grid[r, c] = EMPTY
                elif v in _QUESTION_IDS:
                    grid[r, c] = QUESTION
                elif v in _BRICK_IDS:
                    grid[r, c] = BRICK
                elif v == MARIO or v == ENEMY:
                    pass                        # set by smb_grid, keep as-is
                else:
                    grid[r, c] = SOLID

    @staticmethod
    def _fill_powerup(grid, ram):
        """Place a visible powerup (mushroom / flower / star) on the grid."""
        if ram[_POWERUP_DRAWN] == 0:
            return
        # 0x0039: 0=mushroom, 1=fire flower, 2=star (starman), 3=1-UP mushroom
        kind = int(ram[_POWERUP_KIND])
        if kind == 0:
            grid_val = MUSHROOM
        elif kind == 3:
            grid_val = ONEUP
        elif kind == 1:
            grid_val = FLOWER
        elif kind == 2:
            grid_val = STAR
        else:
            grid_val = MUSHROOM      # unknown — treat as mushroom
        # Same coordinate system as enemies: level_x - x_start → screen col
        x_start = int(ram[0x6D]) * 256 + int(ram[0x86]) - int(ram[0x3AD])
        px = int(ram[_POWERUP_PAGE]) * 256 + int(ram[_POWERUP_X_SCREEN]) - x_start
        py = int(ram[_POWERUP_Y_SCREEN])
        col = (px + 8) // 16
        # Fire flower Y register points one tile above its resting position;
        # other powerups (mushroom, star) use the same anchor as enemies.
        y_offset = 16 if grid_val == FLOWER else 0
        row = (py + 8 - 32 + y_offset) // 16
        if 0 <= row < VISIBLE_ROWS and 0 <= col < VISIBLE_COLS:
            grid[row, col] = grid_val


class AugmentedRAMObservation(gym.ObservationWrapper):
    """Extend the 13x16 grid with a scalar feature vector.

    Wraps RAMGridObservation (or any wrapper that produces a (13, 16) Box).
    Returns a Dict observation compatible with SB3 MultiInputPolicy:
      {
        "grid":    float32 (13, 16)
        "scalars": float32 (NUM_SCALARS_BATCH1,)
      }

    Scalar features — all normalised to [-1, 1] or [0, 1]:
      x_speed, y_velocity, float_state, power_state, inv_timer, game_timer
    """

    def __init__(self, env):
        super().__init__(env)
        grid_space = env.observation_space  # Box(13, 16)
        self.observation_space = spaces.Dict({
            "grid": grid_space,
            "scalars": spaces.Box(
                low=-1.0, high=1.0,
                shape=(NUM_SCALARS_BATCH1,),
                dtype=np.float32,
            ),
        })

    def observation(self, obs):
        ram = self.unwrapped.ram
        return {"grid": obs, "scalars": self._extract_scalars(ram)}

    @staticmethod
    def _extract_scalars(ram):
        # --- x speed (signed byte) ---
        x = int(ram[_MARIO_X_SPEED])
        if x > 127:
            x -= 256
        x_speed = float(np.clip(x / 40.0, -1.0, 1.0))

        # --- y velocity (signed byte) ---
        y = int(ram[_MARIO_Y_VEL])
        if y > 127:
            y -= 256
        y_vel = float(np.clip(y / 8.0, -1.0, 1.0))

        # --- float state (0–3) ---
        float_state = int(ram[_MARIO_FLOAT]) / 3.0

        # --- power state (0=small, 1=big, 2+=fire) ---
        power = min(int(ram[_MARIO_POWER]), 2) / 2.0

        # --- invincibility timer (0–255) ---
        inv = int(ram[_MARIO_INV_TIMER]) / 255.0

        # --- game timer (BCD → integer, max ~400) ---
        timer = (
            int(ram[_TIMER_HUNDREDS]) * 100
            + int(ram[_TIMER_TENS]) * 10
            + int(ram[_TIMER_ONES])
        ) / 400.0

        return np.array([x_speed, y_vel, float_state, power, inv, timer],
                        dtype=np.float32)


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
    env_id="SuperMarioBros-1-1-v0",
    skip=4,
    n_stack=4,
    n_skip=1,
    flatten=False,
    augment=False,
):
    """Create a symbolic-observation Super Mario Bros environment.

    Parameters
    ----------
    env_id : str
        gym-super-mario-bros environment id.
    skip : int
        Number of frames to repeat each action (SkipFrame wrapper).
    n_stack : int
        Number of grid frames to stack (temporal information).
        Ignored when augment=True (scalars already provide velocity info).
        Set to 1 to disable stacking.
    n_skip : int
        Gap between stacked frames. Ignored when augment=True.
    flatten : bool
        If True, flatten the observation to 1-D (for MlpPolicy).
        Ignored when augment=True.
    augment : bool
        If True, return a Dict observation {"grid": (13,16), "scalars": (N,)}
        for use with SB3 MultiInputPolicy.  Frame stacking and flattening are
        disabled in this mode — the scalar features already encode velocity.

    Returns
    -------
    gym.Env
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=skip)
    env = RAMGridObservation(env)

    if augment:
        env = AugmentedRAMObservation(env)
        return env

    if n_stack > 1:
        env = FrameStackGrid(env, n_stack=n_stack, n_skip=n_skip)

    if flatten:
        env = FlattenGrid(env)

    return env
