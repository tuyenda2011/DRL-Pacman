from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.algorithms.dqn.model import QNetwork
from src.algorithms.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            torch.manual_seed(seed)

        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.sync_target_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_capacity, seed=seed)

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        current_q = self.q_network(states).gather(1, actions).squeeze(1)
        target_q = self._compute_target_q(rewards, next_states, dones)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def _compute_target_q(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            return rewards + self.discount_factor * next_q * (1.0 - dones)

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), model_path)

    def load(self, path: str | Path) -> None:
        self.q_network.load_state_dict(torch.load(Path(path), map_location=self.device))
        self.sync_target_network()
