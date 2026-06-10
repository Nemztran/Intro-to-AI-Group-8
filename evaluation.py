class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return self.action_space.sample()


def success_from_info(info):
    return bool(info.get("success", info.get("is_success", False)))


def evaluate_policy(env, select_action, episodes=10, max_episode_steps=100, name="Agent", print_episodes=True, seed=None):
    if seed is not None:
        env.action_space.seed(seed)

    total_reward = 0.0
    total_steps = 0
    successes = 0

    for i_episode in range(episodes):
        episode_reward = 0.0
        episode_steps = 0
        done = False
        episode_seed = seed + i_episode if seed is not None else None
        state, info = env.reset(seed=episode_seed)
        episode_success = success_from_info(info)

        while not done and episode_steps < max_episode_steps:
            action = select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            episode_reward += reward
            episode_success = episode_success or success_from_info(info)
            done = terminated or truncated or episode_success
            state = next_state

        total_reward += episode_reward
        total_steps += episode_steps
        successes += int(episode_success)

        if print_episodes:
            print(
                f"{name} Episode: {i_episode + 1}/{episodes}, "
                f"success: {episode_success}, "
                f"steps: {episode_steps}, "
                f"reward: {round(episode_reward, 2)}"
            )

    metrics = {
        "name": name,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes > 0 else 0.0,
        "avg_steps": total_steps / episodes if episodes > 0 else 0.0,
        "avg_reward": total_reward / episodes if episodes > 0 else 0.0,
    }
    print_metrics(metrics)
    return metrics


def print_metrics(metrics):
    print(
        f"{metrics['name']} summary: "
        f"success_rate={metrics['success_rate']:.2%}, "
        f"avg_steps={metrics['avg_steps']:.2f}, "
        f"avg_reward={metrics['avg_reward']:.2f}"
    )


def print_comparison_table(results):
    print("\nEvaluation comparison")
    print(f"{'Agent':<18} {'Success Rate':>14} {'Avg Steps':>12} {'Avg Reward':>12}")
    print("-" * 60)
    for metrics in results:
        print(
            f"{metrics['name']:<18} "
            f"{metrics['success_rate']:>13.2%} "
            f"{metrics['avg_steps']:>12.2f} "
            f"{metrics['avg_reward']:>12.2f}"
        )
