from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

from src.pacman_env.grid_world import MiniPacmanEnv, State
from src.training.logging_utils import MetricRow

EVAL_FIELDNAMES = [
    "episode",
    "eval_episodes",
    "avg_reward",
    "avg_steps",
    "win_rate",
    "avg_food_eaten",
    "avg_completion_rate",
    "wins",
    "caught",
    "timeout",
    "best_reward",
    "best_completion_rate",
]

StateInput = State | object
StateAdapter = Callable[[MiniPacmanEnv, State], StateInput]
ActionSelector = Callable[[StateInput], int]


def run_epsilon_zero_evaluation(
    *,
    training_episode: int,
    env: MiniPacmanEnv,
    agent: object,
    episodes: int,
    state_adapter: StateAdapter,
) -> MetricRow:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rewards: list[float] = []
    steps: list[int] = []
    food_eaten_values: list[int] = []
    completion_rates: list[float] = []
    event_counts = {"win": 0, "caught": 0, "timeout": 0}

    with _temporary_epsilon(agent, 0.0):
        select_action = _agent_action_selector(agent)
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            event = "timeout"
            step = 0

            for step in range(1, env.max_steps + 1):
                action = select_action(state_adapter(env, state))
                result = env.step(action)
                state = result.state
                total_reward += result.reward
                event = str(result.info["event"])
                if result.done:
                    break

            food_eaten = len(env.food_positions) - env.food_mask.bit_count()
            completion_rate = food_eaten / len(env.food_positions)
            rewards.append(total_reward)
            steps.append(step)
            food_eaten_values.append(food_eaten)
            completion_rates.append(completion_rate)
            if event in event_counts:
                event_counts[event] += 1

    return {
        "episode": training_episode,
        "eval_episodes": episodes,
        "avg_reward": round(sum(rewards) / episodes, 4),
        "avg_steps": round(sum(steps) / episodes, 4),
        "win_rate": round(event_counts["win"] / episodes, 6),
        "avg_food_eaten": round(sum(food_eaten_values) / episodes, 4),
        "avg_completion_rate": round(sum(completion_rates) / episodes, 6),
        "wins": event_counts["win"],
        "caught": event_counts["caught"],
        "timeout": event_counts["timeout"],
        "best_reward": round(max(rewards), 4),
        "best_completion_rate": round(max(completion_rates), 6),
    }


def should_evaluate_episode(episode: int, total_episodes: int, eval_interval: int) -> bool:
    return eval_interval > 0 and (episode == total_episodes or episode % eval_interval == 0)


def build_eval_env(args: object) -> MiniPacmanEnv:
    eval_seed = getattr(args, "eval_seed", None)
    return MiniPacmanEnv(
        layout_name=getattr(args, "layout", "medium"),
        max_steps=getattr(args, "max_steps"),
        max_lives=getattr(args, "max_lives", 3),
        ghost_count=getattr(args, "ghost_count"),
        ghost_chase_probability=getattr(args, "ghost_chase_probability"),
        seed=eval_seed,
    )


def print_evaluation_log(algorithm: str, row: MetricRow) -> None:
    print(
        f"[{algorithm} eval] "
        f"episode {int(row['episode']):>4} | "
        f"avg_reward={float(row['avg_reward']):>8.2f} | "
        f"win_rate={float(row['win_rate']):.2%} | "
        f"complete={float(row['avg_completion_rate']):.2%} | "
        f"food={float(row['avg_food_eaten']):.2f}"
    )


def _agent_action_selector(agent: object) -> ActionSelector:
    select_action = getattr(agent, "select_action", None)
    if not callable(select_action):
        raise TypeError("agent must expose a callable select_action method")
    return select_action


@contextmanager
def _temporary_epsilon(agent: object, value: float) -> Iterator[None]:
    if not hasattr(agent, "epsilon"):
        yield
        return

    original = getattr(agent, "epsilon")
    setattr(agent, "epsilon", value)
    try:
        yield
    finally:
        setattr(agent, "epsilon", original)
