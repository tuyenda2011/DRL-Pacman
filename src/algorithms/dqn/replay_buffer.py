from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        # Cached list view, rebuilt lazily when buffer is modified.
        self._cached_list: list[Transition] | None = None

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=action,
                reward=reward,
                next_state=np.asarray(next_state, dtype=np.float32),
                done=done,
            )
        )
        # Invalidate the cache whenever the buffer is modified.
        self._cached_list = None

    def sample(self, batch_size: int) -> list[Transition]:
        # Rebuild the list view only when the buffer has changed (push invalidates it).
        # This avoids an O(n) copy on every single update step.
        if self._cached_list is None:
            self._cached_list = list(self.buffer)
        indices = self.rng.sample(range(len(self._cached_list)), batch_size)
        return [self._cached_list[i] for i in indices]

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.buffer.maxlen,
            "buffer": list(self.buffer),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.buffer = deque(state["buffer"], maxlen=state["capacity"])
        self.rng.setstate(state["rng_state"])
        self._cached_list = None

    def __len__(self) -> int:
        return len(self.buffer)
