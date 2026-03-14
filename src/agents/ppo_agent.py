"""PPO agent using Stable-Baselines3."""

from stable_baselines3 import PPO
from src.config import PPOConfig, PPO_DEFAULTS


def make_ppo_agent(env, config: PPOConfig = None, tensorboard_log="logs/"):
    """Create a PPO agent for the given environment."""
    if config is None:
        config = PPO_DEFAULTS

    model = PPO(
        policy=config.policy,
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    return model
