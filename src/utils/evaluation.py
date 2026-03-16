"""Evaluation utilities for trained agents."""

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_agent(model, env, n_episodes=30, deterministic=True):
    """Evaluate a trained agent and return detailed metrics.

    Returns:
        dict with keys: mean_reward, std_reward, mean_length, flag_rate,
                        episode_rewards, episode_lengths
    """
    episode_rewards = []
    episode_lengths = []
    flags_reached = []

    for _ in range(n_episodes):
        reward, length, flag = run_episode(model, env, deterministic=deterministic)
        episode_rewards.append(reward)
        episode_lengths.append(length)
        flags_reached.append(flag)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "flag_rate": np.mean(flags_reached),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def run_episode(model, env, deterministic=True, render=False):
    """Run a single episode and return (total_reward, length, flag_reached)."""
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result
    total_reward = 0.0
    length = 0
    done = False
    flag = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_result = env.step(int(action))
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        total_reward += float(reward)
        length += 1
        if render:
            env.render()
        if isinstance(info, dict) and info.get("flag_get", False):
            flag = True

    return total_reward, length, flag
