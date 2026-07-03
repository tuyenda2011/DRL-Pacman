from __future__ import annotations

import pickle
import re
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.algorithms.dqn.agent import DQNAgent
    from src.algorithms.q_learning.agent import QLearningAgent
    from src.pacman_env.grid_world import MiniPacmanEnv


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = latest_checkpoint_path(path)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path


def latest_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        return checkpoint_path

    candidates = _checkpoint_candidates(checkpoint_path)
    if candidates:
        return max(candidates, key=_checkpoint_episode)
    return checkpoint_path


def save_q_learning_checkpoint(
    agent: QLearningAgent,
    args: Namespace,
    env: MiniPacmanEnv,
    episode: int,
    elapsed_sec: float,
    win_count: int,
) -> Path:
    checkpoint = {
        "episode": episode,
        "elapsed_sec": elapsed_sec,
        "win_count": win_count,
        "metadata": {
            "algorithm": "Q-learning",
            "layout": getattr(args, "layout", "medium"),
            "state_size": len(env._state()),
            "ghost_count": args.ghost_count,
            "ghost_chase_probability": args.ghost_chase_probability,
            "max_steps": args.max_steps,
        },
        "epsilon": agent.epsilon,
        "rng_state": agent.rng.getstate(),
        "q_table": dict(agent.q_table),
    }
    checkpoint_path = episode_checkpoint_path(args.checkpoint_path, episode)
    _write_checkpoint(checkpoint_path, checkpoint)
    return checkpoint_path


def load_q_learning_checkpoint(agent: QLearningAgent, path: str | Path) -> dict[str, Any]:
    checkpoint = _read_checkpoint(path)
    agent.q_table = defaultdict(
        lambda: np.zeros(agent.action_size, dtype=np.float32),
        checkpoint["q_table"],
    )
    agent.epsilon = checkpoint["epsilon"]
    agent.rng.setstate(checkpoint["rng_state"])
    return checkpoint


def save_deep_q_checkpoint(
    agent: DQNAgent,
    args: Namespace,
    algorithm: str,
    env: MiniPacmanEnv,
    episode: int,
    global_step: int,
    elapsed_sec: float,
    win_count: int,
) -> Path:
    checkpoint = {
        "episode": episode,
        "global_step": global_step,
        "elapsed_sec": elapsed_sec,
        "win_count": win_count,
        "metadata": {
            "algorithm": algorithm,
            "layout": getattr(args, "layout", "medium"),
            "state_size": env.vector_size,
            "action_size": env.action_size,
            "ghost_count": args.ghost_count,
            "ghost_chase_probability": args.ghost_chase_probability,
            "max_steps": args.max_steps,
            "hidden_size": args.hidden_size,
        },
        "epsilon": agent.epsilon,
        "rng_state": agent.rng.getstate(),
        "q_network": agent.q_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "replay_buffer": agent.replay_buffer.state_dict(),
    }
    checkpoint_path = episode_checkpoint_path(args.checkpoint_path, episode)
    _write_checkpoint(checkpoint_path, checkpoint)
    return checkpoint_path


def episode_checkpoint_path(path: str | Path, episode: int) -> Path:
    checkpoint_path = Path(path)
    suffix = checkpoint_path.suffix or ".pkl"
    stem = checkpoint_path.stem
    stem = re.sub(r"_ep\d+$", "", stem)
    return checkpoint_path.with_name(f"{stem}_ep{episode:04d}{suffix}")


def _checkpoint_candidates(path: Path) -> list[Path]:
    suffix = path.suffix or ".pkl"
    stem = re.sub(r"_ep\d+$", "", path.stem)
    return sorted(path.parent.glob(f"{stem}_ep*{suffix}"))


def _checkpoint_episode(path: Path) -> int:
    match = re.search(r"_ep(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def load_deep_q_checkpoint(agent: DQNAgent, path: str | Path) -> dict[str, Any]:
    checkpoint = _read_checkpoint(path)
    agent.q_network.load_state_dict(checkpoint["q_network"])
    agent.target_network.load_state_dict(checkpoint["target_network"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    agent.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    agent.epsilon = checkpoint["epsilon"]
    agent.rng.setstate(checkpoint["rng_state"])
    return checkpoint


def _write_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as file:
        pickle.dump(checkpoint, file)


def _read_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        return pickle.load(file)
