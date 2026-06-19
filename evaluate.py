import os
import csv
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent

gym.register_envs(gymnasium_robotics)

ENV_NAME = "PointMaze_UMaze-v3"
EVAL_EPISODES = 50
HIDDEN_SIZE = 512
ALPHA = 0.12
LEARNING_RATE = 0.0001
EXPLORATION_SCALING_FACTOR = 1.5
INTRINSIC_REWARD_CLIP = 1.0

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

MEDIUM_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
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

MAZES = {
    "LARGE_MAZE": (LARGE_MAZE, 500),
    "MEDIUM_MAZE": (MEDIUM_MAZE, 500),
    "UNSEEN_TEST_MAZE": (UNSEEN_TEST_MAZE, 500),
}


def make_env(maze_map, max_episode_steps):
    env = gym.make(
        ENV_NAME,
        max_episode_steps=max_episode_steps,
        maze_map=maze_map,
        render_mode=None,
    )
    return RoboGymObservationWrapper(env)


def success_from_info(info):
    return bool(info.get("success", info.get("is_success", False)))


def run_agent(env, select_action_fn, episodes, max_episode_steps):
    rewards = []
    steps_list = []
    successes = []

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_success = success_from_info(info)

        while not done and ep_steps < max_episode_steps:
            action = select_action_fn(state)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            step_success = success_from_info(info)
            ep_success = ep_success or step_success
            done = terminated or truncated or step_success

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        successes.append(int(ep_success))

    return {
        "rewards": rewards,
        "steps": steps_list,
        "successes": successes,
    }


def compute_metrics(name, maze_name, data):
    rewards = np.array(data["rewards"])
    steps = np.array(data["steps"])
    successes = np.array(data["successes"])

    return {
        "agent": name,
        "maze": maze_name,
        "success_rate": np.mean(successes),
        "avg_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "median_reward": np.median(rewards),
        "avg_steps": np.mean(steps),
        "std_steps": np.std(steps),
        "episodes": len(rewards),
    }


def print_results(all_metrics):
    print("\n" + "=" * 90)
    print("EVALUATION RESULTS")
    print("=" * 90)

    header = f"{'Agent':<20} {'Maze':<18} {'Success%':>9} {'Avg Reward':>11} {'Std Reward':>11} {'Avg Steps':>10} {'Median R':>9}"
    print(header)
    print("-" * 90)

    for m in all_metrics:
        print(
            f"{m['agent']:<20} {m['maze']:<18} "
            f"{m['success_rate']*100:>8.1f}% "
            f"{m['avg_reward']:>11.2f} "
            f"{m['std_reward']:>11.2f} "
            f"{m['avg_steps']:>10.2f} "
            f"{m['median_reward']:>9.2f}"
        )

    print("=" * 90)


def save_csv(all_metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["agent", "maze", "success_rate", "avg_reward", "std_reward",
              "min_reward", "max_reward", "median_reward", "avg_steps", "std_steps", "episodes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nResults saved to {path}")


def main():
    all_metrics = []

    for maze_name, (maze_map, max_steps) in MAZES.items():
        print(f"\n--- Evaluating on {maze_name} ({EVAL_EPISODES} episodes) ---")
        env = make_env(maze_map, max_steps)
        obs, _ = env.reset()
        obs_size = obs.shape[0]

        # Random Agent
        print(f"Running Random Agent on {maze_name}...")
        random_data = run_agent(
            env,
            lambda s: env.action_space.sample(),
            EVAL_EPISODES,
            max_steps,
        )
        all_metrics.append(compute_metrics("Random Agent", maze_name, random_data))

        # SAC + Curiosity Agent
        print(f"Running SAC+Curiosity on {maze_name}...")
        agent = Agent(
            obs_size,
            env.action_space,
            gamma=0.99,
            tau=0.005,
            alpha=ALPHA,
            target_update_interval=1,
            hidden_size=HIDDEN_SIZE,
            learning_rate=LEARNING_RATE,
            exploration_scaling_factor=EXPLORATION_SCALING_FACTOR,
            intrinsic_reward_clip=INTRINSIC_REWARD_CLIP,
            use_curiosity=True,
            checkpoint_dir="checkpoints",
        )
        agent.load_checkpoint(evaluate=True)

        sac_data = run_agent(
            env,
            lambda s: agent.select_action(s, evaluate=True),
            EVAL_EPISODES,
            max_steps,
        )
        all_metrics.append(compute_metrics("SAC + Curiosity", maze_name, sac_data))

        env.close()

    print_results(all_metrics)
    save_csv(all_metrics, "results/evaluation_results.csv")


if __name__ == "__main__":
    main()
