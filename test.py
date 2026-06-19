import gymnasium as gym
import gymnasium_robotics
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
from evaluation import RandomAgent, evaluate_policy, print_comparison_table

gym.register_envs(gymnasium_robotics)

ENV_NAME = "PointMaze_UMaze-v3"
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.12
TARGET_UPDATE_INTERVAL = 1
HIDDEN_SIZE = 512
LEARNING_RATE = 0.0001
EXPLORATION_SCALING_FACTOR = 1.5
INTRINSIC_REWARD_CLIP = 1.0
CHECKPOINT_DIR = "checkpoints"
EVAL_EPISODES = 50
EVAL_MAX_EPISODE_STEPS = 400
RENDER_MODE = None
PRINT_EPISODE_DETAILS = False

LARGE_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


if __name__ == "__main__":
    env = gym.make(
        ENV_NAME,
        max_episode_steps=EVAL_MAX_EPISODE_STEPS,
        render_mode=RENDER_MODE,
        maze_map=LARGE_MAZE,
    )
    env = RoboGymObservationWrapper(env)
    obs, info = env.reset()
    obs_size = obs.shape[0]
    agent = Agent(
        obs_size,
        env.action_space,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        exploration_scaling_factor=EXPLORATION_SCALING_FACTOR,
        intrinsic_reward_clip=INTRINSIC_REWARD_CLIP,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    agent.load_checkpoint(evaluate=True)
    sac_metrics = agent.test(
        env=env,
        episodes=EVAL_EPISODES,
        max_episode_steps=EVAL_MAX_EPISODE_STEPS,
        name="SAC Agent",
        print_episodes=PRINT_EPISODE_DETAILS,
    )

    random_agent = RandomAgent(env.action_space)
    random_metrics = evaluate_policy(
        env=env,
        select_action=random_agent.select_action,
        episodes=EVAL_EPISODES,
        max_episode_steps=EVAL_MAX_EPISODE_STEPS,
        name="Random Agent",
        print_episodes=PRINT_EPISODE_DETAILS,
    )

    print_comparison_table([sac_metrics, random_metrics])
    env.close()
