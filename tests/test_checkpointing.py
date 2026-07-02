from argparse import Namespace

import numpy as np
import torch

from src.algorithms.dqn.agent import DQNAgent
from src.algorithms.q_learning.agent import QLearningAgent
from src.pacman_env.grid_world import MiniPacmanEnv
from src.training.checkpointing import (
    episode_checkpoint_path,
    load_deep_q_checkpoint,
    load_q_learning_checkpoint,
    resolve_checkpoint_path,
    save_deep_q_checkpoint,
    save_q_learning_checkpoint,
)


def test_q_learning_checkpoint_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "q_checkpoint.pkl"
    args = Namespace(
        checkpoint_path=checkpoint_path,
        ghost_count=3,
        ghost_chase_probability=0.35,
        max_steps=500,
    )
    agent = QLearningAgent(action_size=4, epsilon=0.8, seed=1)
    agent.learn((1, 1, 0), 2, 10.0, (1, 2, 0), False)
    agent.decay_epsilon()

    saved_path = save_q_learning_checkpoint(
        agent,
        args,
        episode=12,
        elapsed_sec=3.5,
        win_count=2,
    )

    restored = QLearningAgent(action_size=4, epsilon=1.0, seed=99)
    assert saved_path == tmp_path / "q_checkpoint_ep0012.pkl"
    assert episode_checkpoint_path(checkpoint_path, 500) == tmp_path / "q_checkpoint_ep0500.pkl"
    assert resolve_checkpoint_path(checkpoint_path) == saved_path

    checkpoint = load_q_learning_checkpoint(restored, saved_path)

    assert checkpoint["episode"] == 12
    assert checkpoint["win_count"] == 2
    assert checkpoint["metadata"]["algorithm"] == "Q-learning"
    assert restored.epsilon == agent.epsilon
    assert restored.q_table[(1, 1, 0)].tolist() == agent.q_table[(1, 1, 0)].tolist()


def test_deep_q_checkpoint_round_trip_restores_training_state(tmp_path) -> None:
    env = MiniPacmanEnv(max_steps=5, ghost_count=2, seed=1)
    checkpoint_path = tmp_path / "dqn_checkpoint.pkl"
    args = Namespace(
        checkpoint_path=checkpoint_path,
        ghost_count=2,
        ghost_chase_probability=1.0,
        hidden_size=8,
        layout="medium",
        max_steps=5,
    )
    agent = DQNAgent(
        state_size=env.vector_size,
        action_size=env.action_size,
        hidden_size=8,
        batch_size=2,
        replay_capacity=10,
        epsilon=0.4,
        seed=1,
        device="cpu",
    )
    agent.store_transition(
        np.zeros(env.vector_size, dtype=np.float32),
        1,
        2.0,
        np.ones(env.vector_size, dtype=np.float32),
        False,
    )
    agent.store_transition(
        np.ones(env.vector_size, dtype=np.float32),
        2,
        -1.0,
        np.zeros(env.vector_size, dtype=np.float32),
        True,
    )
    agent.update()

    saved_path = save_deep_q_checkpoint(
        agent,
        args,
        "DQN",
        env,
        episode=7,
        global_step=12,
        elapsed_sec=3.25,
        win_count=1,
    )

    restored = DQNAgent(
        state_size=env.vector_size,
        action_size=env.action_size,
        hidden_size=8,
        batch_size=2,
        replay_capacity=10,
        epsilon=1.0,
        seed=99,
        device="cpu",
    )
    checkpoint = load_deep_q_checkpoint(restored, saved_path)

    assert saved_path == tmp_path / "dqn_checkpoint_ep0007.pkl"
    assert checkpoint["episode"] == 7
    assert checkpoint["global_step"] == 12
    assert checkpoint["metadata"]["algorithm"] == "DQN"
    assert restored.epsilon == agent.epsilon
    assert len(restored.replay_buffer) == len(agent.replay_buffer)
    for name, value in agent.q_network.state_dict().items():
        assert torch.equal(restored.q_network.state_dict()[name], value)
    for name, value in agent.target_network.state_dict().items():
        assert torch.equal(restored.target_network.state_dict()[name], value)
    assert len(restored.optimizer.state_dict()["state"]) == len(agent.optimizer.state_dict()["state"])
