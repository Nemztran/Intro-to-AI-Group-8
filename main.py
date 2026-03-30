import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from model import *
gym.register_envs(gymnasium_robotics)

if __name__ == "__main__":
    env_name = "PointMaze_UMaze-v3"
    max_episode_steps = 100

    STRAIGHT_MAZE = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        maze_map=STRAIGHT_MAZE,
        render_mode='human'
    )
    env = RoboGymObservationWrapper(env)
    obs, info = env.reset()
    critic = Critic(8, 2, 256)
    env.close()