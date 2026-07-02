from src.algorithms.q_learning.agent import QLearningAgent
from src.pacman_env.grid_world import MiniPacmanEnv
from src.training.evaluation import (
    EVAL_FIELDNAMES,
    run_epsilon_zero_evaluation,
    should_evaluate_episode,
)


def test_evaluation_forces_epsilon_zero_then_restores_it() -> None:
    env = MiniPacmanEnv(max_steps=5, ghost_count=1, ghost_chase_probability=0.0, seed=1)
    agent = QLearningAgent(action_size=env.action_size, epsilon=0.8, seed=1)

    row = run_epsilon_zero_evaluation(
        training_episode=10,
        env=env,
        agent=agent,
        episodes=3,
        state_adapter=lambda _env, state: state,
    )

    assert agent.epsilon == 0.8
    assert row["episode"] == 10
    assert row["eval_episodes"] == 3
    assert set(row) == set(EVAL_FIELDNAMES)
    assert row["wins"] + row["caught"] + row["timeout"] <= 3


def test_final_episode_is_evaluated_even_when_not_on_interval() -> None:
    assert should_evaluate_episode(episode=10, total_episodes=10, eval_interval=6)
    assert should_evaluate_episode(episode=6, total_episodes=10, eval_interval=6)
    assert not should_evaluate_episode(episode=5, total_episodes=10, eval_interval=6)
    assert not should_evaluate_episode(episode=10, total_episodes=10, eval_interval=0)
