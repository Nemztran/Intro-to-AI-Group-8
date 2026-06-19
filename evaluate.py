import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent

gym.register_envs(gymnasium_robotics)

# Maze dùng để train (đã thấy trong quá trình train)
LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
              [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
              [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
              [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
              [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# Maze độ khó trung bình, chưa từng train
MEDIUM_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
               [1, 0, 0, 0, 1, 0, 0, 1],
               [1, 0, 1, 0, 1, 0, 1, 1],
               [1, 0, 1, 0, 0, 0, 0, 1],
               [1, 0, 1, 1, 1, 1, 0, 1],
               [1, 0, 0, 0, 0, 0, 0, 1],
               [1, 1, 1, 1, 1, 1, 1, 1]]

# Maze hoàn toàn mới, chỉ để test khả năng tổng quát hóa của agent
UNSEEN_TEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                     [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                     [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                     [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

MAZES = {
    "LARGE_MAZE": LARGE_MAZE,
    "MEDIUM_MAZE": MEDIUM_MAZE,
    "UNSEEN_TEST_MAZE": UNSEEN_TEST_MAZE,
}


def make_env(maze_map, max_episode_steps=500):
    env = gym.make(
        "PointMaze_UMaze-v3",
        max_episode_steps=max_episode_steps,
        maze_map=maze_map,
        render_mode=None,
    )
    return RoboGymObservationWrapper(env)


def run_random_agent(env, episodes, max_episode_steps):
    rewards = []
    for _ in range(episodes):
        env.reset()
        done = False
        steps = 0
        ep_reward = 0
        while not done and steps < max_episode_steps:
            action = env.action_space.sample()
            _, reward, done, _, _ = env.step(action)
            ep_reward += reward
            steps += 1
        rewards.append(ep_reward)
    return rewards


def run_sac_agent(env, agent, episodes, max_episode_steps):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        while not done and steps < max_episode_steps:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
            steps += 1
        rewards.append(ep_reward)
    return rewards


def main(episodes=20, max_episode_steps=500, hidden_size=512, alpha=0.12,
         learning_rate=0.0001, exploration_scaling_factor=1.5):
    results = {}

    for maze_name, maze_map in MAZES.items():
        env = make_env(maze_map, max_episode_steps)
        obs_size = env.observation_space.shape[0]

        random_rewards = run_random_agent(env, episodes, max_episode_steps)

        agent = Agent(
            obs_size, env.action_space, gamma=0.99, tau=0.005, alpha=alpha,
            target_update_interval=1, hidden_size=hidden_size,
            learning_rate=learning_rate,
            exploration_scaling_factor=exploration_scaling_factor,
        )
        agent.load_checkpoint(evaluate=True)
        sac_curiosity_rewards = run_sac_agent(env, agent, episodes, max_episode_steps)

        env.close()

        results[maze_name] = {
            "Random Agent": random_rewards,
            "SAC + Curiosity": sac_curiosity_rewards,
        }

    print("\n=== Bang so sanh reward trung binh ===")
    header = f"{'Maze':<18} {'Random Agent':>15} {'SAC + Curiosity':>17}"
    print(header)
    print("-" * len(header))
    for maze_name, agents in results.items():
        row = f"{maze_name:<18}"
        for _, rewards in agents.items():
            row += f"{np.mean(rewards):>17.2f}"
        print(row)


if __name__ == "__main__":
    main()
