# isort:skip_file

import gymnasium as gym
import minigrid  # noqa
import gym_minipacman  # noqa


def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env
