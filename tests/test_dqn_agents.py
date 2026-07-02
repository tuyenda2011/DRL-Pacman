import math

import numpy as np
import torch
from torch import nn

from src.algorithms.double_dqn.agent import DoubleDQNAgent
from src.algorithms.dqn.agent import DQNAgent


class FixedQNetwork(nn.Module):
    def __init__(self, values: list[list[float]]) -> None:
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values[: x.shape[0]]


def test_dqn_target_uses_target_network_max_q_value() -> None:
    agent = DQNAgent(state_size=2, action_size=2, discount_factor=0.5, seed=1, device="cpu")
    agent.target_network = FixedQNetwork([[10.0, 20.0], [30.0, 40.0]])

    target = agent._compute_target_q(
        rewards=torch.tensor([1.0, 2.0]),
        next_states=torch.zeros((2, 2)),
        dones=torch.tensor([0.0, 0.0]),
    )

    assert target.tolist() == [11.0, 22.0]


def test_double_dqn_target_selects_with_online_and_evaluates_with_target() -> None:
    agent = DoubleDQNAgent(state_size=2, action_size=2, discount_factor=0.5, seed=1, device="cpu")
    agent.q_network = FixedQNetwork([[0.0, 5.0], [9.0, 1.0]])
    agent.target_network = FixedQNetwork([[10.0, 20.0], [30.0, 40.0]])

    target = agent._compute_target_q(
        rewards=torch.tensor([1.0, 2.0]),
        next_states=torch.zeros((2, 2)),
        dones=torch.tensor([0.0, 0.0]),
    )

    assert target.tolist() == [11.0, 17.0]


def test_dqn_update_runs_one_minibatch_step() -> None:
    agent = DQNAgent(
        state_size=3,
        action_size=2,
        hidden_size=8,
        batch_size=2,
        replay_capacity=10,
        seed=1,
        device="cpu",
    )

    agent.store_transition(
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        0,
        1.0,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        False,
    )
    agent.store_transition(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        1,
        -1.0,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        True,
    )

    loss = agent.update()

    assert loss is not None
    assert math.isfinite(loss)
