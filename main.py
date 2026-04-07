import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from model import *
gym.register_envs(gymnasium_robotics)
from buffer import ReplayBuffer
if __name__ == "__main__":
    replay_buffer_size = 1000000
    episodes = 1000
    warmup_steps = 10000
    batch_size = 69
    gamma = 0.99
    update_per_step = 4
    hidden_size = 256
    tau = 0.99
    alpha = 0.12
    target_update_interval = 1
    max_episode_steps = 100
    learning_rate = 0.0003
    exploration_scaling_factor = 1.0

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
    
    observation, info = env.reset()

    observation_size = observation.shape[0]

    # Khởi tạo agent
    agent = Agent(observation_size, env.action_space, gamma = gamma, tau = tau, alpha = alpha, target_update_interval = target_update_interval, hidden_size = hidden_size, learning_rate = learning_rate, exploration_scaling_factor = exploration_scaling_factor)
    
    memory = ReplayBuffer(replay_buffer_size, input_size = observation_size, action_dim = env.action_space.shape[0])
    

    env.close()
    