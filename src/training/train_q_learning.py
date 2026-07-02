from __future__ import annotations

import argparse
from contextlib import ExitStack
import time
from pathlib import Path
from typing import Any

from src.algorithms.q_learning.agent import QLearningAgent
from src.pacman_env.grid_world import MiniPacmanEnv
from src.training.checkpointing import (
    load_q_learning_checkpoint,
    resolve_checkpoint_path,
    save_q_learning_checkpoint,
)
from src.training.config_utils import parse_args_with_config
from src.training.evaluation import (
    EVAL_FIELDNAMES,
    build_eval_env,
    print_evaluation_log,
    run_epsilon_zero_evaluation,
    should_evaluate_episode,
)
from src.training.logging_utils import (
    MetricRow,
    MetricsCsvLogger,
    append_training_run_log,
    maybe_render_env,
    print_episode_log,
    print_saved_outputs,
    print_training_header,
    read_metric_rows,
    should_log_episode,
)

Q_LEARNING_FIELDNAMES = [
    "episode",
    "reward",
    "steps",
    "epsilon",
    "event",
    "win_rate",
    "food_eaten",
    "completion_rate",
    "elapsed_sec",
    "q_states",
]


def train(args: argparse.Namespace) -> list[MetricRow]:
    env = MiniPacmanEnv(
        layout_name=args.layout,
        max_steps=args.max_steps,
        max_lives=args.max_lives,
        ghost_count=args.ghost_count,
        ghost_chase_probability=args.ghost_chase_probability,
        seed=args.seed,
    )
    agent = QLearningAgent(
        action_size=env.action_size,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    args.food_count = len(env.food_positions)
    args.map_size = f"{env.width}x{env.height}"

    resume_state: dict[str, Any] | None = None
    if args.resume:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
        resume_state = load_q_learning_checkpoint(agent, checkpoint_path)
        print(
            f"Resumed checkpoint from episode {resume_state['episode']}: {checkpoint_path} "
            f"| epsilon={agent.epsilon:.6f} q_states={len(agent.q_table)}"
        )

    print_training_header(
        "Q-learning",
        args,
        [
            "Reset MiniPacmanEnv at the start of each episode.",
            "Choose an action with epsilon-greedy from the Q-table.",
            "Run env.step(action) to receive next_state, reward and done.",
            "Update Q(s,a) using reward + gamma * max Q(next_state).",
            "Decay epsilon, then write episode metrics to CSV.",
        ],
    )

    rows = _resume_metric_rows(args.output, resume_state)
    start_episode = int(resume_state.get("episode", 0)) + 1 if resume_state else 1
    win_count = int(resume_state.get("win_count", _count_wins(rows))) if resume_state else 0
    previous_elapsed = float(resume_state.get("elapsed_sec", 0.0)) if resume_state else 0.0
    started_at = time.perf_counter() - previous_elapsed
    run_status = "completed"

    eval_metrics_rows = _resume_metric_rows(args.eval_output, resume_state) if _evaluation_enabled(args) else []

    with ExitStack() as stack:
        metrics = stack.enter_context(
            MetricsCsvLogger(args.output, Q_LEARNING_FIELDNAMES, existing_rows=rows)
        )
        eval_metrics = None
        if _evaluation_enabled(args):
            eval_metrics = stack.enter_context(
                MetricsCsvLogger(args.eval_output, EVAL_FIELDNAMES, existing_rows=eval_metrics_rows)
            )

        if start_episode > args.episodes:
            print(f"Checkpoint already reached episode {start_episode - 1}; nothing new to train.")
            all_rows = read_metric_rows(args.output)
            append_training_run_log(
                args.history_output,
                algorithm="Q-learning",
                args=args,
                status="skipped",
                start_episode=start_episode,
                final_episode=start_episode - 1,
                final_metrics=all_rows[-1] if all_rows else None,
            )
            return all_rows

        for episode in range(start_episode, args.episodes + 1):
            state = env.reset()
            total_reward = 0.0
            event = "timeout"
            interrupted = False
            step = 0

            try:
                for step in range(1, args.max_steps + 1):
                    action = agent.select_action(state)
                    result = env.step(action)
                    agent.learn(state, action, result.reward, result.state, result.done)
                    state = result.state
                    total_reward += result.reward
                    event = str(result.info["event"])
                    if result.done:
                        break
            except KeyboardInterrupt:
                interrupted = True
                event = "interrupted"

            agent.decay_epsilon()
            if event == "win":
                win_count += 1
            win_rate = win_count / episode
            food_eaten = len(env.food_positions) - env.food_mask.bit_count()
            completion_rate = food_eaten / len(env.food_positions)
            elapsed_sec = time.perf_counter() - started_at
            row = {
                "episode": episode,
                "reward": round(total_reward, 4),
                "steps": step,
                "epsilon": round(agent.epsilon, 6),
                "event": event,
                "win_rate": round(win_rate, 6),
                "food_eaten": food_eaten,
                "completion_rate": round(completion_rate, 6),
                "elapsed_sec": round(elapsed_sec, 3),
                "q_states": len(agent.q_table),
            }
            metrics.write(row)

            if should_log_episode(episode, args.episodes, args.log_interval, event):
                print_episode_log(
                    "Q-learning",
                    episode,
                    args.episodes,
                    total_reward,
                    step,
                    agent.epsilon,
                    event,
                    extra=(
                        f"win_rate={win_rate:.2%} food={food_eaten}/{len(env.food_positions)} "
                        f"complete={completion_rate:.2%} elapsed={elapsed_sec:.1f}s "
                        f"q_states={len(agent.q_table)}"
                    ),
                )
                maybe_render_env(env.render(), episode, args.render_interval)

            if args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
                checkpoint_path = save_q_learning_checkpoint(agent, args, episode, elapsed_sec, win_count)
                print(f"Checkpoint saved to: {checkpoint_path}")
            if (
                not interrupted
                and eval_metrics
                and should_evaluate_episode(episode, args.episodes, args.eval_interval)
            ):
                eval_row = run_epsilon_zero_evaluation(
                    training_episode=episode,
                    env=build_eval_env(args),
                    agent=agent,
                    episodes=args.eval_episodes,
                    state_adapter=lambda _eval_env, eval_state: eval_state,
                )
                eval_metrics.write(eval_row)
                print_evaluation_log("Q-learning", eval_row)
            if interrupted:
                run_status = "interrupted"
                print("Training interrupted by user. Resume from the latest scheduled checkpoint with --resume.")
                break

    saved_model_output = args.model_output if run_status == "completed" else None
    if saved_model_output is not None:
        model_path = Path(saved_model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(model_path)
    all_rows = read_metric_rows(args.output)
    append_training_run_log(
        args.history_output,
        algorithm="Q-learning",
        args=args,
        status=run_status,
        start_episode=start_episode,
        final_episode=int(all_rows[-1]["episode"]) if all_rows else start_episode - 1,
        final_metrics=all_rows[-1] if all_rows else None,
    )
    print_saved_outputs(args.output, saved_model_output, args.history_output, status=run_status)
    return all_rows


def _resume_metric_rows(output: str, resume_state: dict[str, Any] | None) -> list[MetricRow]:
    if not resume_state:
        return []
    if resume_state.get("rows"):
        return list(resume_state["rows"])
    return read_metric_rows(output, up_to_episode=int(resume_state.get("episode", 0)))


def _count_wins(rows: list[MetricRow]) -> int:
    return sum(1 for row in rows if row.get("event") == "win")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Q-learning on Mini Pacman.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--layout", choices=sorted(MiniPacmanEnv.LAYOUTS), default="medium")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-lives", type=int, default=3)
    parser.add_argument("--ghost-count", type=int, choices=[2, 3], default=3)
    parser.add_argument("--ghost-chase-probability", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--render-interval", type=int, default=0)
    parser.add_argument("--output", default="experiments/metrics/{algorithm}/{run_name}_metrics.csv")
    parser.add_argument("--history-output", default="experiments/history/{algorithm}/{run_name}_history.jsonl")
    parser.add_argument("--model-output", default="models/final/{algorithm}/{run_name}.pkl")
    parser.add_argument(
        "--checkpoint-path",
        default="models/checkpoints/{algorithm}/{run_name}/{run_name}_checkpoint.pkl",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-output", default="experiments/metrics/{algorithm}/{run_name}_eval_metrics.csv")
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--resume", action="store_true")
    return parser


def _evaluation_enabled(args: argparse.Namespace) -> bool:
    return args.eval_interval > 0 and args.eval_episodes > 0


if __name__ == "__main__":
    train(parse_args_with_config(build_parser(), "configs/q_learning/q_learning_lr_001.yaml"))
