import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
from buffer import ReplayBuffer
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
    tau = 0.005
    alpha = 0.12
    target_update_interval = 1
    max_episode_steps = 100 
    learning_rate = 0.0003
    exploration_scaling_factor = 1.0

    env_name = "PointMaze_UMaze-v3"
    max_episode_steps = 100
    replay_buffer_size = 1000000
    episodes = 1000
    batch_size = 64
    updates_per_step = 4
    gamma = 0.99
    tau = 0.005
    alpha = 0.12
    target_update_interval = 1
    hidden_size = 512
    learning_rate = 0.0001
    exploration_scaling_factor = 1.5
    STRAIGHT_MAZE = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps,
        maze_map=STRAIGHT_MAZE,
        render_mode=None
    )
    env = RoboGymObservationWrapper(env)
    obs, info = env.reset()
    obs_size = obs.shape[0]
    #Agent
    agent = Agent(obs_size, env.action_space, gamma=gamma, tau=tau, alpha=alpha, target_update_interval=target_update_interval, hidden_size=hidden_size, learning_rate=learning_rate, exploration_scaling_factor=exploration_scaling_factor)
    memory = ReplayBuffer(replay_buffer_size, input_size=obs_size, n_actions=env.action_space.shape[0])
    agent.train(env=env, env_name=env_name, memory=memory, episodes=100, batch_size=batch_size,
                updates_per_step=updates_per_step, summary_writer_name=f"straight_maze={alpha}_lr={learning_rate}_hs={hidden_size}_a={alpha}_phase_1",
                max_episode_steps=max_episode_steps)
    

    #Training phase 2
    LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    max_episode_steps_phase_2 = 500
    env = gym.make(
        env_name,
        max_episode_steps=max_episode_steps_phase_2,
        maze_map=LARGE_MAZE,
        render_mode=None
    )
    env = RoboGymObservationWrapper(env)
    agent.train(env=env, env_name=env_name, memory=memory, episodes=3000, batch_size=batch_size,
                updates_per_step=updates_per_step, summary_writer_name=f"large_maze={alpha}_lr={learning_rate}_hs={hidden_size}_a={alpha}_phase_2",
                max_episode_steps=max_episode_steps_phase_2)

    env.close()
