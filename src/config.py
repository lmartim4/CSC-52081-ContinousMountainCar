"""Centralized hyperparameters for all experiments."""

from dataclasses import dataclass, field
from typing import List


# --- Environment ---
ENV_ID = "SuperMarioBros-1-1-v3"
SIMPLE_MOVEMENT = True  # Use simplified action space (7 actions)

# --- Pixel observation settings ---
FRAME_SHAPE = (84, 84)
FRAME_STACK = 4
GRAYSCALE = True

# --- Symbolic observation settings ---
GRID_SHAPE = (13, 16)  # rows x cols for the symbolic grid


@dataclass
class DQNConfig:
    """Hyperparameters for DQN agent."""
    learning_rate: float = 0.00025
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    batch_size: int = 32
    gamma: float = 0.9
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.1
    train_freq: int = 3
    total_timesteps: int = 2_000_000
    policy: str = "CnnPolicy"


@dataclass
class PPOConfig:
    """Hyperparameters for PPO agent."""
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.9
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 2_000_000
    policy: str = "MlpPolicy"


@dataclass
class EvalConfig:
    """Evaluation settings."""
    n_eval_episodes: int = 30
    deterministic: bool = True
    render: bool = False


# Default configs
DQN_DEFAULTS = DQNConfig()
PPO_DEFAULTS = PPOConfig()
EVAL_DEFAULTS = EvalConfig()
