import argparse
import csv
import os
import random
from statistics import mean, stdev

import numpy as np
import torch

from agent import Agent
from evaluation import RandomAgent, evaluate_policy
from main import (
    ALPHA,
    GAMMA,
    HIDDEN_SIZE,
    INTRINSIC_REWARD_CLIP,
    LARGE_MAZE,
    LEARNING_RATE,
    TARGET_UPDATE_INTERVAL,
    TAU,
    make_env,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC checkpoints over multiple seeds.")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seed list.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument(
        "--agent",
        action="append",
        help="Agent spec in NAME=CHECKPOINT_DIR format. Can be passed multiple times.",
    )
    parser.add_argument("--no-random", action="store_true", help="Skip Random Agent baseline.")
    parser.add_argument("--output-csv", default="results/evaluation_seeds.csv")
    parser.add_argument("--summary-csv", default="results/evaluation_summary.csv")
    return parser.parse_args()


def parse_seeds(seed_text):
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def parse_agent_specs(agent_specs):
    if not agent_specs:
        return [("SAC Agent", "checkpoints")]

    parsed = []
    for spec in agent_specs:
        if "=" in spec:
            name, checkpoint_dir = spec.split("=", 1)
            parsed.append((name.strip(), checkpoint_dir.strip()))
        else:
            parsed.append((os.path.basename(spec.rstrip("/\\")) or "SAC Agent", spec))
    return parsed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_agent(env, checkpoint_dir):
    obs, _ = env.reset()
    return Agent(
        obs.shape[0],
        env.action_space,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        exploration_scaling_factor=0.0,
        intrinsic_reward_clip=INTRINSIC_REWARD_CLIP,
        use_curiosity=False,
        checkpoint_dir=checkpoint_dir,
    )


def evaluate_sac_agent(name, checkpoint_dir, seed, episodes, max_episode_steps):
    set_seed(seed)
    env = make_env(LARGE_MAZE, max_episode_steps)
    try:
        agent = create_agent(env, checkpoint_dir)
        agent.load_checkpoint(evaluate=True)
        metrics = agent.test(
            env=env,
            episodes=episodes,
            max_episode_steps=max_episode_steps,
            name=name,
            print_episodes=False,
            seed=seed,
        )
    finally:
        env.close()
    return {**metrics, "seed": seed, "checkpoint_dir": checkpoint_dir}


def evaluate_random_agent(seed, episodes, max_episode_steps):
    set_seed(seed)
    env = make_env(LARGE_MAZE, max_episode_steps)
    try:
        random_agent = RandomAgent(env.action_space)
        metrics = evaluate_policy(
            env=env,
            select_action=random_agent.select_action,
            episodes=episodes,
            max_episode_steps=max_episode_steps,
            name="Random Agent",
            print_episodes=False,
            seed=seed,
        )
    finally:
        env.close()
    return {**metrics, "seed": seed, "checkpoint_dir": ""}


def write_csv(path, rows):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = ["name", "seed", "episodes", "success_rate", "avg_steps", "avg_reward", "checkpoint_dir"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["name"], []).append(row)

    summary_rows = []
    for name, group in grouped.items():
        success_rates = [row["success_rate"] for row in group]
        avg_steps = [row["avg_steps"] for row in group]
        avg_rewards = [row["avg_reward"] for row in group]
        summary_rows.append(
            {
                "name": name,
                "seeds": len(group),
                "success_rate_mean": mean(success_rates),
                "success_rate_std": stdev(success_rates) if len(success_rates) > 1 else 0.0,
                "avg_steps_mean": mean(avg_steps),
                "avg_steps_std": stdev(avg_steps) if len(avg_steps) > 1 else 0.0,
                "avg_reward_mean": mean(avg_rewards),
                "avg_reward_std": stdev(avg_rewards) if len(avg_rewards) > 1 else 0.0,
            }
        )
    return summary_rows


def write_summary_csv(path, rows):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = [
        "name",
        "seeds",
        "success_rate_mean",
        "success_rate_std",
        "avg_steps_mean",
        "avg_steps_std",
        "avg_reward_mean",
        "avg_reward_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\nMulti-seed evaluation summary")
    print(
        f"{'Agent':<20} {'Seeds':>6} {'Success Rate':>24} "
        f"{'Avg Steps':>22} {'Avg Reward':>22}"
    )
    print("-" * 98)
    for row in rows:
        print(
            f"{row['name']:<20} "
            f"{row['seeds']:>6} "
            f"{row['success_rate_mean']:>10.2%} +/- {row['success_rate_std']:<8.2%} "
            f"{row['avg_steps_mean']:>9.2f} +/- {row['avg_steps_std']:<8.2f} "
            f"{row['avg_reward_mean']:>9.2f} +/- {row['avg_reward_std']:<8.2f}"
        )


if __name__ == "__main__":
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    agent_specs = parse_agent_specs(args.agent)

    rows = []
    for seed in seeds:
        for name, checkpoint_dir in agent_specs:
            rows.append(evaluate_sac_agent(name, checkpoint_dir, seed, args.episodes, args.max_episode_steps))
        if not args.no_random:
            rows.append(evaluate_random_agent(seed, args.episodes, args.max_episode_steps))

    summary_rows = summarize(rows)
    write_csv(args.output_csv, rows)
    write_summary_csv(args.summary_csv, summary_rows)
    print_summary(summary_rows)
    print(f"\nSaved detailed rows to {args.output_csv}")
    print(f"Saved summary rows to {args.summary_csv}")
