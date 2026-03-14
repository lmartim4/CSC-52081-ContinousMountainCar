"""DQN agent using Stable-Baselines3."""

from stable_baselines3 import DQN
from src.config import DQNConfig, DQN_DEFAULTS


def make_dqn_agent(env, config: DQNConfig = None, tensorboard_log="logs/"):
    """Create a DQN agent for the given environment.

    Uses CnnPolicy for pixel observations and MlpPolicy for symbolic.
    """
    if config is None:
        config = DQN_DEFAULTS

    model = DQN(
        policy=config.policy,
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        train_freq=config.train_freq,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    return model
