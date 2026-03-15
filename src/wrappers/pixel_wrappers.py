"""Wrappers for pixel-based observations: grayscale, resize, frame stacking."""

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
from gym import spaces
from collections import deque

from src.config import FRAME_SHAPE, FRAME_STACK, GRAYSCALE, ENV_ID, SIMPLE_MOVEMENT as USE_SIMPLE


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB observation to grayscale."""

    def __init__(self, env):
        super().__init__(env)
        h, w = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = np.mean(obs, axis=2, keepdims=True).astype(np.uint8)
        return gray


class ResizeObservation(gym.ObservationWrapper):
    """Resize observation to a target shape."""

    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        channels = self.observation_space.shape[2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shape[0], shape[1], channels), dtype=np.uint8
        )

    def observation(self, obs):
        import cv2
        resized = cv2.resize(obs, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        return resized


class FrameStack(gym.Wrapper):
    """Stack the last k frames as a single observation."""

    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c * k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2)


class SkipFrame(gym.Wrapper):
    """Return every skip-th frame, repeating the action in between."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return obs, {}

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                obs, reward, done, info = result
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, False, info


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize pixel values to [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32) / 255.0


def make_pixel_env(env_id=ENV_ID, frame_shape=FRAME_SHAPE, frame_stack=FRAME_STACK,
                   grayscale=GRAYSCALE, skip=4, normalize=True):
    """Create a fully wrapped pixel-based Mario environment."""
    env = gym_super_mario_bros.make(env_id)
    actions = SIMPLE_MOVEMENT if USE_SIMPLE else COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = SkipFrame(env, skip=skip)
    if grayscale:
        env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=frame_shape)
    if normalize:
        env = NormalizeObservation(env)
    env = FrameStack(env, k=frame_stack)
    return env
