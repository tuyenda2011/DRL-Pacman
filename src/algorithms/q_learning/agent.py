from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Hashable

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: defaultdict[Hashable, np.ndarray] = defaultdict(
            lambda: np.zeros(action_size, dtype=np.float32)
        )
        self.rng = random.Random(seed)

    def select_action(self, state: Hashable) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def learn(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
    ) -> None:
        current_q = self.q_table[state][action]
        next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.discount_factor * next_q
        self.q_table[state][action] += self.learning_rate * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as file:
            pickle.dump(dict(self.q_table), file)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as file:
            data = pickle.load(file)
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_size, dtype=np.float32),
            data,
        )
