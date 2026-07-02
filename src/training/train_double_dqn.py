from __future__ import annotations

from typing import Any

try:
    from src.algorithms.double_dqn.agent import DoubleDQNAgent
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise SystemExit(
        "PyTorch is not installed. Install dependencies first: "
        "pip install -r requirements.txt"
    ) from exc

from src.pacman_env.grid_world import MiniPacmanEnv
from src.training.checkpointing import load_deep_q_checkpoint, resolve_checkpoint_path
from src.training.config_utils import parse_args_with_config
from src.training.logging_utils import append_training_run_log, print_saved_outputs
from src.training.train_dqn import build_parser, run_deep_q_training


def main() -> None:
    parser = build_parser()
    parser.description = "Train Double DQN on Mini Pacman."
    parser.set_defaults(
        output="experiments/metrics/{algorithm}/{run_name}_metrics.csv",
        eval_output="experiments/metrics/{algorithm}/{run_name}_eval_metrics.csv",
        model_output="models/final/{algorithm}/{run_name}.pt",
        checkpoint_path="models/checkpoints/{algorithm}/{run_name}/{run_name}_checkpoint.pkl",
    )
    args = parse_args_with_config(parser, "configs/double_dqn/double_dqn_lr_001.yaml")
    args.algorithm_name = "Double DQN"

    env = MiniPacmanEnv(
        layout_name=args.layout,
        max_steps=args.max_steps,
        max_lives=args.max_lives,
        ghost_count=args.ghost_count,
        ghost_chase_probability=args.ghost_chase_probability,
        seed=args.seed,
    )
    agent = DoubleDQNAgent(
        state_size=env.vector_size,
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
    start_episode = int(resume_state.get("episode", 0)) + 1 if resume_state else 1
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


if __name__ == "__main__":
    main()
