import argparse
import os
import random

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
from buffer import ReplayBuffer

gym.register_envs(gymnasium_robotics)

ENV_NAME = "PointMaze_UMaze-v3"
REPLAY_BUFFER_SIZE = 1_000_000
BATCH_SIZE = 64
GAMMA = 0.99
UPDATES_PER_STEP = 4
HIDDEN_SIZE = 512
TAU = 0.005
ALPHA = 0.12
TARGET_UPDATE_INTERVAL = 1
LEARNING_RATE = 0.0001
EXPLORATION_SCALING_FACTOR = 1.5
INTRINSIC_REWARD_CLIP = 1.0
PHASE_1_EPISODES = 100
PHASE_2_EPISODES = 3000
PHASE_3_EPISODES = 1000
PHASE_1_MAX_EPISODE_STEPS = 100
PHASE_2_MAX_EPISODE_STEPS = 500
PHASE_3_MAX_EPISODE_STEPS = 500
WARMUP_EPISODES = 20

STRAIGHT_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

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

UNSEEN_TEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def make_env(maze_map, max_episode_steps, render_mode=None):
    env = gym.make(
        ENV_NAME,
        max_episode_steps=max_episode_steps,
        maze_map=maze_map,
        render_mode=render_mode,
    )
    return RoboGymObservationWrapper(env)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC agent on PointMaze.")
    parser.add_argument("--phase1-episodes", type=int, default=PHASE_1_EPISODES)
    parser.add_argument("--phase2-episodes", type=int, default=PHASE_2_EPISODES)
    parser.add_argument("--phase3-episodes", type=int, default=PHASE_3_EPISODES)
    parser.add_argument("--skip-phase1", action="store_true", help="Skip phase 1 training.")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip phase 2 training.")
    parser.add_argument("--disable-curiosity", action="store_true", help="Train SAC without intrinsic curiosity reward.")
    parser.add_argument("--disable-auto-alpha", action="store_true", help="Use fixed alpha instead of auto-tuning.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory used to save model checkpoints.")
    parser.add_argument("--experiment-name", default=None, help="Optional suffix for TensorBoard run names.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible training.")
    return parser.parse_args()


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    use_curiosity = not args.disable_curiosity
    auto_alpha = not args.disable_auto_alpha
    exploration_scaling_factor = EXPLORATION_SCALING_FACTOR if use_curiosity else 0.0
    intrinsic_reward_clip = INTRINSIC_REWARD_CLIP if use_curiosity else None
    experiment_name = args.experiment_name or ("sac_curiosity" if use_curiosity else "sac_only")

    env = make_env(STRAIGHT_MAZE, PHASE_1_MAX_EPISODE_STEPS)
    obs, info = env.reset(seed=args.seed)
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
        exploration_scaling_factor=exploration_scaling_factor,
        intrinsic_reward_clip=intrinsic_reward_clip,
        use_curiosity=use_curiosity,
        checkpoint_dir=args.checkpoint_dir,
        auto_alpha=auto_alpha,
    )
    agent.load_checkpoint()

    memory = ReplayBuffer(REPLAY_BUFFER_SIZE, input_size=obs_size, n_actions=env.action_space.shape[0])
    buffer_path = os.path.join(args.checkpoint_dir, 'replay_buffer.npz')
    memory.load(buffer_path)

    # Phase 1 - Straight Maze
    if not args.skip_phase1:
        agent.train(
            env=env,
            env_name=ENV_NAME,
            memory=memory,
            episodes=args.phase1_episodes,
            batch_size=BATCH_SIZE,
            updates_per_step=UPDATES_PER_STEP,
            summary_writer_name=f"{experiment_name}_straight_maze_lr={LEARNING_RATE}_hs={HIDDEN_SIZE}_phase_1",
            max_episode_steps=PHASE_1_MAX_EPISODE_STEPS,
            warmup_episodes=WARMUP_EPISODES,
            seed=args.seed,
        )
    env.close()

    # Phase 2 - Large Maze
    if not args.skip_phase2:
        env = make_env(LARGE_MAZE, PHASE_2_MAX_EPISODE_STEPS)
        agent.train(
            env=env,
            env_name=ENV_NAME,
            memory=memory,
            episodes=args.phase2_episodes,
            batch_size=BATCH_SIZE,
            updates_per_step=UPDATES_PER_STEP,
            summary_writer_name=f"{experiment_name}_large_maze_lr={LEARNING_RATE}_hs={HIDDEN_SIZE}_phase_2",
            max_episode_steps=PHASE_2_MAX_EPISODE_STEPS,
            warmup_episodes=WARMUP_EPISODES,
            seed=args.seed + args.phase1_episodes if args.seed is not None else None,
        )
        env.close()
