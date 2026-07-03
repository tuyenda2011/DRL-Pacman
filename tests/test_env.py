import pytest

from src.pacman_env.grid_world import MiniPacmanEnv


def test_reset_returns_tabular_state_and_vector() -> None:
    env = MiniPacmanEnv(seed=1)
    state = env.reset()
    vector = env.state_vector(state)

    assert env.layout_name == "medium"
    assert env.width == 15
    assert env.height == 15
    assert len(env.food_positions) == 62
    assert len(env.bonus_fruit_positions) == 2
    assert len(env.ghost_positions) == 3
    assert len(state) == 10
    assert env.lives_remaining == 3
    assert env.bonus_fruit_mask == 0b11
    assert vector.shape == (env.vector_size,)
    assert env.vector_size == 2 + (2 * env.ghost_count) + 1 + len(env.food_positions) + 14
    assert env.action_size == 4
    assert env.ghost_names == ("Blinky", "Pinky", "Inky")


def test_environment_can_use_two_ghosts() -> None:
    env = MiniPacmanEnv(ghost_count=2, seed=1)
    state = env.reset()

    assert len(env.ghost_positions) == 2
    assert len(state) == 8
    assert env.ghost_names == ("Blinky", "Pinky")
    assert env.state_vector(state).shape == (env.vector_size,)


def test_medium_layout_supports_up_to_three_ghosts() -> None:
    env = MiniPacmanEnv(ghost_count=3, seed=1)
    state = env.reset()

    assert env.width == 15
    assert env.height == 15
    assert len(env.ghost_positions) == 3
    assert len(state) == 10


def test_medium_layout_rejects_four_ghosts_as_too_hard() -> None:
    with pytest.raises(ValueError, match="ghost_count must be between 1 and 3"):
        MiniPacmanEnv(ghost_count=4, seed=1)


def test_wall_blocks_movement() -> None:
    env = MiniPacmanEnv(seed=1)
    start_state = env.reset()
    result = env.step(2)

    assert result.state[0] == start_state[0]
    assert result.state[1] == start_state[1]
    assert result.info["event"] == "wall"
    assert result.reward < -1.0


def test_ghost_pathfinder_returns_valid_action() -> None:
    env = MiniPacmanEnv(seed=1)
    env.reset()

    start = (1, 11)
    target = (1, 1)
    action = env._next_action_toward(start, target, env._valid_actions(start))

    assert action in env.ACTIONS
    assert env._move(start, action) not in env.walls


def test_directional_ghost_prefers_best_maze_action() -> None:
    env = MiniPacmanEnv(ghost_chase_probability=1.0, seed=1)
    env.reset()

    start = (1, 11)
    target = (1, 1)
    valid_actions = env._valid_actions(start)
    action = env._directional_ghost_action(start, target, valid_actions)
    best_distance = min(
        env._maze_distance(env._move(start, candidate), target)
        for candidate in valid_actions
    )

    assert env._maze_distance(env._move(start, action), target) == best_distance


def test_pacman_style_ghost_targets_use_chase_personalities() -> None:
    env = MiniPacmanEnv(ghost_count=2, seed=1)
    env.reset()
    env.steps = 30
    env.pacman_pos = (7, 7)
    env.last_pacman_action = 1

    assert env._ghost_target(0, env.ghost_positions[0]) == (7, 7)
    assert env._ghost_target(1, env.ghost_positions[1]) == (7, 9)


def test_ghosts_use_scatter_mode_before_chase_mode() -> None:
    env = MiniPacmanEnv(ghost_count=2, seed=1)
    env.reset()

    assert env._ghost_mode() == "scatter"
    env.steps = 30
    assert env._ghost_mode() == "chase"


def test_ghost_lookahead_uses_last_pacman_action() -> None:
    env = MiniPacmanEnv(seed=1)
    env.reset()

    env.step(1)

    assert env.last_pacman_action == 1


def test_survival_reward_encourages_moving_away_from_ghost() -> None:
    env = MiniPacmanEnv(ghost_count=1, seed=1)
    env.reset()
    env.pacman_pos = (1, 1)
    env.ghost_positions = [(1, 4)]

    assert env._survival_reward(previous_ghost_distance=2) == 0.07


def test_survival_reward_penalizes_moving_closer_to_nearby_ghost() -> None:
    env = MiniPacmanEnv(ghost_count=1, seed=1)
    env.reset()
    env.pacman_pos = (1, 1)
    env.ghost_positions = [(1, 3)]

    assert env._survival_reward(previous_ghost_distance=3) == -0.08


def test_food_progress_reward_guides_toward_nearest_food() -> None:
    env = MiniPacmanEnv(ghost_count=1, seed=1)
    env.reset()
    env.pacman_pos = (1, 2)

    assert env._food_progress_reward(previous_food_distance=2) == 0.03


def test_food_progress_reward_penalizes_moving_away_from_nearest_food() -> None:
    env = MiniPacmanEnv(ghost_count=1, seed=1)
    env.reset()
    env.pacman_pos = (13, 5)

    assert env._food_progress_reward(previous_food_distance=0) == -0.03


def test_bonus_fruit_can_be_eaten_once_for_reward() -> None:
    env = MiniPacmanEnv(ghost_count=1, ghost_chase_probability=0.0, seed=1)
    env.reset()
    fruit = env.bonus_fruit_positions[0]
    env.pacman_pos = (fruit[0], fruit[1] + 1)
    env.ghost_positions = [(13, 13)]

    result = env.step(3)

    assert env.pacman_pos == fruit
    assert result.info["bonus_fruit_eaten"] is True
    assert result.info["bonus_fruits_remaining"] == 1
    assert result.reward >= env.BONUS_FRUIT_REWARD

    env.pacman_pos = (fruit[0], fruit[1] + 1)
    env.ghost_positions = [(13, 13)]
    result = env.step(3)

    assert result.info["bonus_fruit_eaten"] is False
    assert result.info["bonus_fruits_remaining"] == 1
    assert result.reward < env.BONUS_FRUIT_REWARD


def test_render_marks_available_bonus_fruit() -> None:
    env = MiniPacmanEnv(seed=1)
    env.reset()
    row, col = env.bonus_fruit_positions[0]

    assert env.render().splitlines()[row][col] == "C"


def test_timeout_keeps_terminal_event_and_penalty() -> None:
    env = MiniPacmanEnv(max_steps=1, ghost_count=1, ghost_chase_probability=0.0, seed=1)
    env.reset()

    result = env.step(1)

    assert result.done is True
    assert result.info["event"] == "timeout"
    assert result.reward < 0.0


def test_caught_loses_life_and_resets_actors_without_ending_episode() -> None:
    env = MiniPacmanEnv(ghost_count=1, seed=1)
    env.reset()
    env.pacman_pos = (1, 1)
    env.ghost_positions = [(1, 2)]
    env.food_mask &= ~1
    food_mask_before_caught = env.food_mask

    result = env.step(1)

    assert result.done is False
    assert result.info["event"] == "life_lost"
    assert result.info["lives"] == 2
    assert env.lives_remaining == 2
    assert env.pacman_pos == env.pacman_start
    assert env.ghost_positions == env.ghost_starts[:1]
    assert env.food_mask == food_mask_before_caught


def test_caught_on_last_life_ends_episode() -> None:
    env = MiniPacmanEnv(ghost_count=1, seed=1)
    env.reset()
    env.lives_remaining = 1
    env.pacman_pos = (1, 1)
    env.ghost_positions = [(1, 2)]

    result = env.step(1)

    assert result.done is True
    assert result.info["event"] == "caught"
    assert result.info["lives"] == 0


def test_life_loss_on_last_step_finishes_as_timeout() -> None:
    env = MiniPacmanEnv(max_steps=1, ghost_count=1, seed=1)
    env.reset()
    env.pacman_pos = (1, 1)
    env.ghost_positions = [(1, 2)]

    result = env.step(1)

    assert result.done is True
    assert result.info["event"] == "timeout"
    assert result.info["lives"] == 2
    assert env.pacman_pos == env.pacman_start
