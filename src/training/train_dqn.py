from __future__ import annotations

import argparse
from contextlib import ExitStack
import time
from typing import Any

try:
    from src.algorithms.dqn.agent import DQNAgent
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise SystemExit(
        "PyTorch is not installed. Install dependencies first: "
        "pip install -r requirements.txt"
    ) from exc

from src.pacman_env.grid_world import MiniPacmanEnv
from src.training.checkpointing import (
    load_deep_q_checkpoint,
    resolve_checkpoint_path,
    save_deep_q_checkpoint,
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

DQN_FIELDNAMES = [
    "episode",
    "reward",
    "steps",
    "epsilon",
    "loss",
    "event",
    "win_rate",
    "food_eaten",
    "completion_rate",
    "elapsed_sec",
    "buffer_size",
    "global_step",
]


def train(args: argparse.Namespace) -> list[MetricRow]:
    args.algorithm_name = "DQN"
    env = MiniPacmanEnv(
        layout_name=args.layout,
        max_steps=args.max_steps,
        max_lives=args.max_lives,
        ghost_count=args.ghost_count,
        ghost_chase_probability=args.ghost_chase_probability,
        seed=args.seed,
    )
    state_size = env.vector_size
    agent = DQNAgent(
        state_size=state_size,
        action_size=env.action_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    resume_state: dict[str, Any] | None = None
    if args.resume:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
        resume_state = load_deep_q_checkpoint(agent, checkpoint_path)
        print(
            f"Resumed checkpoint from episode {resume_state['episode']} "
            f"at global_step {resume_state['global_step']}: {checkpoint_path} "
            f"| epsilon={agent.epsilon:.6f} buffer={len(agent.replay_buffer)}"
        )

    rows = run_deep_q_training(env, agent, args, resume_state=resume_state)
    start_episode = _start_episode_from_resume(resume_state)
    status = getattr(args, "training_status", "completed")
    saved_model_output = args.model_output if status == "completed" else None
    if saved_model_output is not None:
        agent.save(saved_model_output)
    append_training_run_log(
        args.history_output,
        algorithm=args.algorithm_name,
        args=args,
        status=status,
        start_episode=start_episode,
        final_episode=int(rows[-1]["episode"]) if rows else start_episode - 1,
        final_metrics=rows[-1] if rows else None,
    )
    print_saved_outputs(args.output, saved_model_output, args.history_output, status=status)
    return rows


def run_deep_q_training(
    env: MiniPacmanEnv,
    agent: DQNAgent,
    args: argparse.Namespace,
    resume_state: dict[str, Any] | None = None,
) -> list[MetricRow]:
    algorithm = getattr(args, "algorithm_name", "DQN")
    args.food_count = len(env.food_positions)
    args.map_size = f"{env.width}x{env.height}"
    target_step = (
        "Use the online network to choose next action and the target network to evaluate it."
        if algorithm == "Double DQN"
        else "Use the target network max Q(next_state) value to build the TD target."
    )
    print_training_header(
        algorithm,
        args,
        [
            "Reset MiniPacmanEnv and convert the tabular state to a numeric vector.",
            "Choose an action with epsilon-greedy from the online Q-network.",
            "Store (state, action, reward, next_state, done) in replay buffer.",
            f"Wait until replay buffer has at least learning_starts={args.learning_starts} transitions.",
            "Sample a mini-batch from replay buffer and update the online network.",
            target_step,
            "Sync the target network every target_update_interval environment steps.",
            "Decay epsilon per environment step (after learning_starts), write metrics, save checkpoints.",
        ],
    )

    rows = _resume_metric_rows(args.output, resume_state)
    global_step = int(resume_state.get("global_step", 0)) if resume_state else 0
    start_episode = int(resume_state.get("episode", 0)) + 1 if resume_state else 1
    win_count = int(resume_state.get("win_count", _count_wins(rows))) if resume_state else 0
    previous_elapsed = float(resume_state.get("elapsed_sec", 0.0)) if resume_state else 0.0
    started_at = time.perf_counter() - previous_elapsed
    args.training_status = "completed"

    eval_metrics_rows = _resume_metric_rows(args.eval_output, resume_state) if _evaluation_enabled(args) else []

    with ExitStack() as stack:
        metrics = stack.enter_context(MetricsCsvLogger(args.output, DQN_FIELDNAMES, existing_rows=rows))
        eval_metrics = None
        if _evaluation_enabled(args):
            eval_metrics = stack.enter_context(
                MetricsCsvLogger(args.eval_output, EVAL_FIELDNAMES, existing_rows=eval_metrics_rows)
            )

        if start_episode > args.episodes:
            print(f"Checkpoint already reached episode {start_episode - 1}; nothing new to train.")
            args.training_status = "skipped"
            return read_metric_rows(args.output)

        for episode in range(start_episode, args.episodes + 1):
            state = env.reset()
            state_vector = env.state_vector(state)
            total_reward = 0.0
            losses: list[float] = []
            event = "timeout"
            interrupted = False
            step = 0

            try:
                for step in range(1, args.max_steps + 1):
                    global_step += 1
                    action = agent.select_action(state_vector)
                    result = env.step(action)
                    next_vector = env.state_vector(result.state)
                    agent.store_transition(state_vector, action, result.reward, next_vector, result.done)

                    loss = agent.update() if global_step >= args.learning_starts else None
                    if loss is not None:
                        losses.append(loss)
                        # Decay epsilon per step after warmup, proportional to true experience.
                        agent.decay_epsilon()

                    if global_step % args.target_update_interval == 0:
                        agent.sync_target_network()

                    state_vector = next_vector
                    total_reward += result.reward
                    event = str(result.info["event"])
                    if result.done:
                        break
            except KeyboardInterrupt:
                interrupted = True
                event = "interrupted"

            # Epsilon is now decayed per step; no per-episode decay needed.
            if event == "win":
                win_count += 1
            win_rate = win_count / episode
            food_eaten = len(env.food_positions) - env.food_mask.bit_count()
            completion_rate = food_eaten / len(env.food_positions)
            elapsed_sec = time.perf_counter() - started_at
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            row = {
                "episode": episode,
                "reward": round(total_reward, 4),
                "steps": step,
                "epsilon": round(agent.epsilon, 6),
                "loss": round(avg_loss, 6),
                "event": event,
                "win_rate": round(win_rate, 6),
                "food_eaten": food_eaten,
                "completion_rate": round(completion_rate, 6),
                "elapsed_sec": round(elapsed_sec, 3),
                "buffer_size": len(agent.replay_buffer),
                "global_step": global_step,
            }
            metrics.write(row)

            if should_log_episode(episode, args.episodes, args.log_interval, event):
                warmup_text = ""
                if global_step < args.learning_starts:
                    warmup_text = f" warmup={global_step}/{args.learning_starts}"
                print_episode_log(
                    algorithm,
                    episode,
                    args.episodes,
                    total_reward,
                    step,
                    agent.epsilon,
                    event,
                    loss=avg_loss,
                    extra=(
                        f"win_rate={win_rate:.2%} food={food_eaten}/{len(env.food_positions)} "
                        f"complete={completion_rate:.2%} elapsed={elapsed_sec:.1f}s "
                        f"buffer={len(agent.replay_buffer)} step={global_step}{warmup_text}"
                    ),
                )
                maybe_render_env(env.render(), episode, args.render_interval)

            if args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
                checkpoint_path = save_deep_q_checkpoint(
                    agent,
                    args,
                    algorithm,
                    env,
                    episode,
                    global_step,
                    elapsed_sec,
                    win_count,
                )
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
                    state_adapter=lambda eval_env, eval_state: eval_env.state_vector(eval_state),
                )
                eval_metrics.write(eval_row)
                print_evaluation_log(algorithm, eval_row)
            if interrupted:
                args.training_status = "interrupted"
                print("Training interrupted by user. Resume from the latest scheduled checkpoint with --resume.")
                break

    all_rows = read_metric_rows(args.output)
    return all_rows


def save_rows(rows: list[MetricRow], output: str) -> None:
    with MetricsCsvLogger(output, DQN_FIELDNAMES, existing_rows=rows):
        pass


def _resume_metric_rows(output: str, resume_state: dict[str, Any] | None) -> list[MetricRow]:
    if not resume_state:
        return []
    if resume_state.get("rows"):
        return list(resume_state["rows"])
    return read_metric_rows(output, up_to_episode=int(resume_state.get("episode", 0)))


def _count_wins(rows: list[MetricRow]) -> int:
    return sum(1 for row in rows if row.get("event") == "win")


def _start_episode_from_resume(resume_state: dict[str, Any] | None) -> int:
    return int(resume_state.get("episode", 0)) + 1 if resume_state else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DQN on Mini Pacman.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=128)
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
    parser.add_argument("--model-output", default="models/final/{algorithm}/{run_name}.pt")
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
    train(parse_args_with_config(build_parser(), "configs/dqn/dqn_lr_001.yaml"))
