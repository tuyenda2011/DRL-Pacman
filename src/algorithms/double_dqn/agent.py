from __future__ import annotations

import torch

from src.algorithms.dqn.agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    def _compute_target_q(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            return rewards + self.discount_factor * next_q * (1.0 - dones)
