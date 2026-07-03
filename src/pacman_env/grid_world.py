from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

Action = int
Position = tuple[int, int]
State = tuple[int, ...]


@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    done: bool
    info: dict[str, object]


class MiniPacmanEnv:
    STEP_PENALTY = -0.1
    FOOD_REWARD = 10.0
    POWER_PELLET_REWARD = 50.0
    BONUS_FRUIT_REWARD = 100.0
    GHOST_EAT_REWARD = 200.0
    WIN_REWARD = 100.0
    CAUGHT_PENALTY = -50.0
    SURVIVAL_REWARD = 0.02
    FOOD_DISTANCE_GAIN_REWARD = 0.03
    FOOD_DISTANCE_LOSS_PENALTY = -0.03
    GHOST_DISTANCE_GAIN_REWARD = 0.05
    GHOST_DISTANCE_LOSS_PENALTY = -0.1
    GHOST_DANGER_DISTANCE = 5
    ACTION_DANGER_DISTANCE = 2
    TIMEOUT_PENALTY = -10.0
    WALL_HIT_PENALTY = -1.0
    FRIGHTENED_STEPS = 35
    FRIGHTENED_GHOST_MOVE_INTERVAL = 2
    GHOST_RESPAWN_WAIT_STEPS = 4

    ACTIONS: dict[Action, Position] = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }
    REVERSE_ACTIONS: dict[Action, Action] = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }
    SCATTER_CHASE_SCHEDULE = (
        ("scatter", 25),
        ("chase", 75),
        ("scatter", 25),
        ("chase", 75),
        ("scatter", 15),
        ("chase", 75),
        ("scatter", 15),
        ("chase", None),
    )
    MAX_GHOSTS = 3
    GHOST_NAMES = ("Blinky", "Pinky", "Inky")
    BONUS_FRUIT_POSITIONS = ((3, 3), (11, 11))

    MEDIUM_LAYOUT = (
        "###############",
        "#..... . .....#",
        "#.##. # # .##.#",
        "#.... # # ....#",
        "# ##   .   ## #",
        "#....     ....#",
        "### ###=### ###",
        "#...# GGG #...#",
        "#   #######   #",
        "#..... . .....#",
        "# #    .    # #",
        "#.. # # # # ..#",
        "#.##. # # .##.#",
        "#  ..  P  ..  #",
        "###############",
    )

    DEFAULT_LAYOUT = MEDIUM_LAYOUT
    LAYOUTS = {
        "medium": MEDIUM_LAYOUT,
    }

    def __init__(
        self,
        layout: Iterable[str] | None = None,
        layout_name: str = "medium",
        max_steps: int = 200,
        ghost_count: int = 3,
        max_lives: int = 3,
        ghost_chase_probability: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if layout is None:
            if layout_name not in self.LAYOUTS:
                choices = ", ".join(sorted(self.LAYOUTS))
                raise ValueError(f"Unknown layout_name {layout_name!r}. Choices: {choices}.")
            layout = self.LAYOUTS[layout_name]
        self.layout_name = layout_name
        self.layout = tuple(layout)
        self.height = len(self.layout)
        self.width = len(self.layout[0])
        self.max_steps = max_steps
        self.ghost_count = ghost_count
        self.max_lives = max_lives
        self.ghost_chase_probability = ghost_chase_probability
        self.rng = random.Random(seed)

        self.walls: set[Position] = set()
        self.ghost_doors: set[Position] = set()
        self.food_positions: list[Position] = []
        self.power_pellet_positions: list[Position] = []
        self.bonus_fruit_positions: list[Position] = []
        self.pacman_start: Position | None = None
        self.ghost_starts: list[Position] = []
        self._parse_layout()
        self.power_pellet_positions = self._select_power_pellet_positions()
        self.bonus_fruit_positions = self._select_bonus_fruit_positions()
        self._validate_layout()

        # Precompute full BFS distance matrix once; eliminates per-step BFS.
        # _dist[src][dst] = shortest maze distance (or manhattan fallback if unreachable).
        self._dist: dict[Position, dict[Position, int]] = self._precompute_distances()

        self.pacman_pos = self.pacman_start
        self.ghost_positions = list(self.ghost_starts[: self.ghost_count])
        self.ghost_returning = [False] * self.ghost_count
        self.ghost_respawn_timers = [0] * self.ghost_count
        self.food_mask = 0
        self.bonus_fruit_mask = 0
        self.frightened_timer = 0
        self.lives_remaining = self.max_lives
        self.steps = 0
        self.last_pacman_action: Action = 1
        self.ghost_last_actions: list[Action | None] = [None] * self.ghost_count

    @property
    def action_size(self) -> int:
        return len(self.ACTIONS)

    @property
    def ghost_names(self) -> tuple[str, ...]:
        return self.GHOST_NAMES[: self.ghost_count]

    @property
    def vector_size(self) -> int:
        return 2 + (4 * self.ghost_count) + 2 + len(self.food_positions) + 14

    def reset(self) -> State:
        self._reset_actor_positions()
        self.food_mask = (1 << len(self.food_positions)) - 1
        self.bonus_fruit_mask = (1 << len(self.bonus_fruit_positions)) - 1
        self.frightened_timer = 0
        self.lives_remaining = self.max_lives
        self.steps = 0
        return self._state()

    def step(self, action: Action) -> StepResult:
        self.steps += 1
        reward = self.STEP_PENALTY
        done = False
        event = "step"
        previous_ghost_distance = self._nearest_ghost_distance()
        previous_food_distance = self._nearest_food_distance()

        self.last_pacman_action = action
        previous_pacman_pos = self.pacman_pos
        self.pacman_pos = self._move(self.pacman_pos, action)
        if self.pacman_pos == previous_pacman_pos:
            reward += self.WALL_HIT_PENALTY
            event = "wall"

        ghosts_eaten = self._eat_ghosts_at_pacman()
        if ghosts_eaten:
            reward += self.GHOST_EAT_REWARD * ghosts_eaten
            event = "ghost_eaten"
        elif self._is_caught():
            return self._handle_caught(reward + self.CAUGHT_PENALTY)

        food_reward, power_pellet_eaten = self._eat_food_reward()
        bonus_fruit_reward = self._eat_bonus_fruit_reward()
        bonus_fruit_eaten = bonus_fruit_reward > 0.0
        if power_pellet_eaten:
            self.frightened_timer = self.FRIGHTENED_STEPS
            event = "power_pellet"
        reward += food_reward + bonus_fruit_reward
        if food_reward == 0.0 and bonus_fruit_reward == 0.0 and self.food_mask != 0:
            reward += self._food_progress_reward(previous_food_distance)
        if self.food_mask == 0:
            return self._finish(
                reward + self.WIN_REWARD,
                "win",
                bonus_fruit_eaten,
                power_pellet_eaten,
                ghosts_eaten,
            )

        self.ghost_positions = self._move_ghosts()
        moved_ghosts_eaten = self._eat_ghosts_at_pacman()
        if moved_ghosts_eaten:
            ghosts_eaten += moved_ghosts_eaten
            reward += self.GHOST_EAT_REWARD * moved_ghosts_eaten
            event = "ghost_eaten"
        elif self._is_caught():
            return self._handle_caught(reward + self.CAUGHT_PENALTY, bonus_fruit_eaten, power_pellet_eaten)

        reward += self._survival_reward(previous_ghost_distance)

        if self.steps >= self.max_steps:
            reward += self.TIMEOUT_PENALTY
            done = True
            event = "timeout"

        if self.frightened_timer > 0 and not power_pellet_eaten:
            self.frightened_timer -= 1

        return StepResult(
            self._state(),
            reward,
            done,
            self._step_info(event, bonus_fruit_eaten, power_pellet_eaten, ghosts_eaten),
        )

    def state_vector(self, state: State | None = None) -> np.ndarray:
        state = state or self._state()
        pacman_row, pacman_col = state[0], state[1]
        ghost_values = state[2 : 2 + (2 * self.ghost_count)]
        ghost_status_start = 2 + (2 * self.ghost_count)
        ghost_returning_values = state[ghost_status_start : ghost_status_start + self.ghost_count]
        ghost_respawn_start = ghost_status_start + self.ghost_count
        ghost_respawn_values = state[ghost_respawn_start : ghost_respawn_start + self.ghost_count]
        frightened_timer = state[-3]
        lives_remaining = state[-2]
        food_mask = state[-1]
        ghost_positions = [
            (ghost_values[index], ghost_values[index + 1])
            for index in range(0, len(ghost_values), 2)
        ]
        dangerous_ghost_positions = [
            ghost_pos
            for index, ghost_pos in enumerate(ghost_positions)
            if ghost_returning_values[index] == 0 and ghost_respawn_values[index] == 0
        ]
        vector = [
            pacman_row / (self.height - 1),
            pacman_col / (self.width - 1),
        ]
        for index, value in enumerate(ghost_values):
            denominator = self.height - 1 if index % 2 == 0 else self.width - 1
            vector.append(value / denominator)
        vector.extend(float(value) for value in ghost_returning_values)
        vector.extend(value / self.GHOST_RESPAWN_WAIT_STEPS for value in ghost_respawn_values)
        vector.append(lives_remaining / self.max_lives)
        vector.append(frightened_timer / self.FRIGHTENED_STEPS)
        vector.extend(1.0 if food_mask & (1 << index) else 0.0 for index in range(len(self.food_positions)))
        vector.extend(
            self._auxiliary_state_features(
                (pacman_row, pacman_col),
                dangerous_ghost_positions,
                food_mask,
                frightened_timer,
            )
        )
        return np.asarray(vector, dtype=np.float32)

    def render(self) -> str:
        cells = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for row, col in self.walls:
            cells[row][col] = "#"
        for row, col in self.ghost_doors:
            cells[row][col] = "="
        for index, (row, col) in enumerate(self.food_positions):
            cells[row][col] = "." if self.food_mask & (1 << index) else " "
        for index, (row, col) in enumerate(self.bonus_fruit_positions):
            if self.bonus_fruit_mask & (1 << index):
                cells[row][col] = "C"
        pacman_row, pacman_col = self.pacman_pos
        cells[pacman_row][pacman_col] = "P"
        for index, (ghost_row, ghost_col) in enumerate(self.ghost_positions):
            if self.ghost_returning[index]:
                cells[ghost_row][ghost_col] = "E"
            elif self.ghost_respawn_timers[index] > 0:
                cells[ghost_row][ghost_col] = "R"
            else:
                cells[ghost_row][ghost_col] = "F" if self.frightened_timer > 0 else "G"
            if (ghost_row, ghost_col) == self.pacman_pos and self._ghost_can_catch(index):
                cells[ghost_row][ghost_col] = "X"
        return "\n".join("".join(row) for row in cells)

    def _parse_layout(self) -> None:
        for row_index, row in enumerate(self.layout):
            if len(row) != self.width:
                raise ValueError("All layout rows must have the same width.")
            for col_index, cell in enumerate(row):
                pos = (row_index, col_index)
                if cell == "#":
                    self.walls.add(pos)
                elif cell == "=":
                    self.ghost_doors.add(pos)
                elif cell == "P":
                    self.pacman_start = pos
                elif cell == "G":
                    self.ghost_starts.append(pos)
                elif cell == ".":
                    self.food_positions.append(pos)

    def _validate_layout(self) -> None:
        if self.pacman_start is None:
            raise ValueError("Layout must contain one P.")
        if self.max_lives < 1:
            raise ValueError("max_lives must be at least 1.")
        max_ghosts = min(self.MAX_GHOSTS, len(self.ghost_starts))
        if not 1 <= self.ghost_count <= max_ghosts:
            raise ValueError(
                f"ghost_count must be between 1 and {max_ghosts} for this layout."
            )
        if len(self.food_positions) > 63:
            raise ValueError("Food mask supports up to 63 food positions.")

    def _select_bonus_fruit_positions(self) -> list[Position]:
        unavailable = {self.pacman_start, *self.ghost_starts}
        return [
            position
            for position in self.BONUS_FRUIT_POSITIONS
            if 0 <= position[0] < self.height
            and 0 <= position[1] < self.width
            and position not in self.walls
            and position not in unavailable
        ]

    def _select_power_pellet_positions(self) -> list[Position]:
        corners = [
            (0, 0),
            (0, self.width - 1),
            (self.height - 1, 0),
            (self.height - 1, self.width - 1),
        ]
        selected: list[Position] = []
        for corner in corners:
            if not self.food_positions:
                break
            pellet = min(
                self.food_positions,
                key=lambda pos: self._manhattan(pos, corner),
            )
            if pellet not in selected:
                selected.append(pellet)
        return selected

    def _state(self) -> State:
        pacman_row, pacman_col = self.pacman_pos
        values: list[int] = [pacman_row, pacman_col]
        for ghost_row, ghost_col in self.ghost_positions:
            values.extend([ghost_row, ghost_col])
        values.extend(1 if returning else 0 for returning in self.ghost_returning)
        values.extend(self.ghost_respawn_timers)
        values.append(self.frightened_timer)
        values.append(self.lives_remaining)
        values.append(self.food_mask)
        return tuple(values)

    def _reset_actor_positions(self) -> None:
        self.pacman_pos = self.pacman_start
        self.ghost_positions = list(self.ghost_starts[: self.ghost_count])
        self.ghost_returning = [False] * self.ghost_count
        self.ghost_respawn_timers = [0] * self.ghost_count
        self.frightened_timer = 0
        self.last_pacman_action = 1
        self.ghost_last_actions = [None] * self.ghost_count

    def _move(self, position: Position, action: Action, *, allow_ghost_door: bool = False) -> Position:
        delta_row, delta_col = self.ACTIONS[action]
        candidate = (position[0] + delta_row, position[1] + delta_col)
        if not (0 <= candidate[0] < self.height and 0 <= candidate[1] < self.width):
            return position
        if candidate in self.walls:
            return position
        if candidate in self.ghost_doors and not allow_ghost_door:
            return position
        return candidate

    def _move_ghosts(self) -> list[Position]:
        next_positions: list[Position] = []
        next_actions: list[Action | None] = []
        for ghost_index, ghost_pos in enumerate(self.ghost_positions):
            next_pos, action = self._move_ghost(ghost_pos, ghost_index)
            next_positions.append(next_pos)
            next_actions.append(action)
        self.ghost_last_actions = next_actions
        return next_positions

    def _move_ghost(self, ghost_pos: Position, ghost_index: int) -> tuple[Position, Action | None]:
        if self.ghost_respawn_timers[ghost_index] > 0:
            self.ghost_respawn_timers[ghost_index] -= 1
            return ghost_pos, None

        if self.ghost_returning[ghost_index]:
            target = self.ghost_starts[ghost_index]
            if ghost_pos == target:
                self.ghost_returning[ghost_index] = False
                self.ghost_respawn_timers[ghost_index] = self.GHOST_RESPAWN_WAIT_STEPS
                return ghost_pos, None
            valid_actions = self._valid_actions(ghost_pos, allow_ghost_door=True)
            if not valid_actions:
                return ghost_pos, None
            action = self._next_action_toward(
                ghost_pos,
                target,
                valid_actions,
                allow_ghost_door=True,
            )
            next_pos = self._move(ghost_pos, action, allow_ghost_door=True)
            if next_pos == target:
                self.ghost_returning[ghost_index] = False
                self.ghost_respawn_timers[ghost_index] = self.GHOST_RESPAWN_WAIT_STEPS
            return next_pos, action

        valid_actions = self._valid_actions(ghost_pos, allow_ghost_door=True)
        if not valid_actions:
            return ghost_pos, None

        if self.frightened_timer > 0:
            if self.steps % self.FRIGHTENED_GHOST_MOVE_INTERVAL != 0:
                return ghost_pos, None
            action = self._frightened_ghost_action(ghost_pos, valid_actions)
            return self._move(ghost_pos, action, allow_ghost_door=True), action

        valid_actions = self._without_reverse_action(ghost_index, valid_actions)

        target = self._ghost_target(ghost_index, ghost_pos)
        action = self._directional_ghost_action(ghost_pos, target, valid_actions)
        return self._move(ghost_pos, action, allow_ghost_door=True), action

    def _frightened_ghost_action(self, ghost_pos: Position, valid_actions: list[Action]) -> Action:
        candidate_distances = [
            self._maze_distance(self._move(ghost_pos, action, allow_ghost_door=True), self.pacman_pos)
            for action in valid_actions
        ]
        best_distance = max(candidate_distances)
        best_actions = [
            action
            for action, distance in zip(valid_actions, candidate_distances)
            if distance == best_distance
        ]
        return self.rng.choice(best_actions)

    def _directional_ghost_action(
        self,
        ghost_pos: Position,
        target: Position,
        valid_actions: list[Action],
    ) -> Action:
        # Arcade-style ghosts choose the legal direction that minimizes maze
        # distance to their current target. ghost_chase_probability can lower
        # determinism for easier experiments, but configs keep it at 1.0.
        candidate_distances = [
            self._maze_distance(self._move(ghost_pos, action, allow_ghost_door=True), target)
            for action in valid_actions
        ]
        best_distance = min(candidate_distances)
        best_actions = [
            action
            for action, distance in zip(valid_actions, candidate_distances)
            if distance == best_distance
        ]

        action_weights = {
            action: (1.0 - self.ghost_chase_probability) / len(valid_actions)
            for action in valid_actions
        }
        for action in best_actions:
            action_weights[action] += self.ghost_chase_probability / len(best_actions)
        return self._choose_weighted_action(action_weights)

    def _valid_actions(self, position: Position, *, allow_ghost_door: bool = False) -> list[Action]:
        return [
            action
            for action in self.ACTIONS
            if self._move(position, action, allow_ghost_door=allow_ghost_door) != position
        ]

    def _ghost_target(self, ghost_index: int, ghost_pos: Position) -> Position:
        if self._ghost_mode() == "scatter":
            return self._scatter_target(ghost_index, ghost_pos)
        # Blinky: direct chase Pacman.
        if ghost_index == 0:
            return self.pacman_pos
        # Pinky: ambush four tiles ahead of Pacman's current direction.
        if ghost_index == 1:
            return self._lookahead_target(4)
        # Inky: use Blinky and a point two tiles ahead to form a vector target.
        if ghost_index == 2:
            return self._inky_target()
        return self.pacman_pos

    def _ghost_mode(self) -> str:
        if self.frightened_timer > 0:
            return "frightened"
        elapsed = self.steps
        for mode, duration in self.SCATTER_CHASE_SCHEDULE:
            if duration is None:
                return mode
            if elapsed <= duration:
                return mode
            elapsed -= duration
        return "chase"

    def _without_reverse_action(self, ghost_index: int, valid_actions: list[Action]) -> list[Action]:
        if ghost_index >= len(self.ghost_last_actions):
            return valid_actions
        last_action = self.ghost_last_actions[ghost_index]
        if last_action is None:
            return valid_actions
        reverse_action = self.REVERSE_ACTIONS[last_action]
        filtered = [action for action in valid_actions if action != reverse_action]
        return filtered or valid_actions

    def _scatter_target(self, ghost_index: int, ghost_pos: Position) -> Position:
        corners = [
            (1, self.width - 2),
            (1, 1),
            (self.height - 2, self.width - 2),
            (self.height - 2, 1),
        ]
        target = corners[ghost_index % len(corners)]
        if target in self.walls:
            return self._nearest_open_corner(ghost_pos)
        return target

    def _inky_target(self) -> Position:
        ahead = self._lookahead_target(2)
        blinky = self.ghost_positions[0] if self.ghost_positions else self.pacman_pos
        target = (
            ahead[0] + (ahead[0] - blinky[0]),
            ahead[1] + (ahead[1] - blinky[1]),
        )
        return self._nearest_open_position(target)

    def _nearest_open_position(self, target: Position) -> Position:
        if target not in self.walls and 0 <= target[0] < self.height and 0 <= target[1] < self.width:
            return target
        return min(
            self._open_positions(),
            key=lambda position: self._manhattan(position, target),
        )

    def _lookahead_target(self, distance: int) -> Position:
        target = self.pacman_pos
        for _ in range(distance):
            candidate = self._move(target, self.last_pacman_action)
            if candidate == target:
                break
            target = candidate
        return target

    def _nearest_open_corner(self, position: Position) -> Position:
        corners = [
            (1, 1),
            (1, self.width - 2),
            (self.height - 2, 1),
            (self.height - 2, self.width - 2),
        ]
        open_corners = [corner for corner in corners if corner not in self.walls]
        return min(open_corners, key=lambda corner: self._manhattan(position, corner))

    def _next_action_toward(
        self,
        start: Position,
        target: Position,
        fallback_actions: list[Action],
        *,
        allow_ghost_door: bool = False,
    ) -> Action:
        if start == target:
            return self.rng.choice(fallback_actions)

        if target in self.walls:
            target = min(
                self._open_positions(),
                key=lambda position: self._manhattan(position, target),
            )

        queue: deque[Position] = deque([start])
        visited: set[Position] = {start}
        first_action: dict[Position, Action] = {}

        while queue:
            position = queue.popleft()
            if position == target and position in first_action:
                return first_action[position]
            for action in self.ACTIONS:
                next_position = self._move(position, action, allow_ghost_door=allow_ghost_door)
                if next_position == position or next_position in visited:
                    continue
                visited.add(next_position)
                first_action[next_position] = first_action.get(position, action)
                queue.append(next_position)

        return min(
            fallback_actions,
            key=lambda action: self._manhattan(
                self._move(start, action, allow_ghost_door=allow_ghost_door),
                target,
            ),
        )

    def _maze_distance(self, start: Position, target: Position) -> int:
        """Return shortest maze distance using the precomputed BFS matrix (O(1))."""
        if start == target:
            return 0
        # Resolve wall targets to the nearest open cell.
        if target in self.walls:
            target = min(
                self._open_positions(),
                key=lambda position: self._manhattan(position, target),
            )
        # O(1) lookup from the precomputed distance matrix.
        src_row = self._dist.get(start)
        if src_row is not None:
            dist = src_row.get(target)
            if dist is not None:
                return dist
        # Fallback: should never happen on a valid open maze position.
        return self._manhattan(start, target)

    def _choose_weighted_action(self, weights: dict[Action, float]) -> Action:
        threshold = self.rng.random()
        cumulative = 0.0
        last_action = next(iter(weights))
        for action, weight in weights.items():
            cumulative += weight
            last_action = action
            if threshold <= cumulative:
                return action
        return last_action

    def _open_positions(self) -> list[Position]:
        return [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in self.walls
        ]

    def _eat_food_reward(self) -> tuple[float, bool]:
        reward = 0.0
        power_pellet_eaten = False
        for index, food_pos in enumerate(self.food_positions):
            if self.pacman_pos == food_pos and self.food_mask & (1 << index):
                self.food_mask &= ~(1 << index)
                reward += self.FOOD_REWARD
                if food_pos in self.power_pellet_positions:
                    power_pellet_eaten = True
                    reward += self.POWER_PELLET_REWARD
        return reward, power_pellet_eaten

    def _eat_bonus_fruit_reward(self) -> float:
        reward = 0.0
        for index, fruit_pos in enumerate(self.bonus_fruit_positions):
            if self.pacman_pos == fruit_pos and self.bonus_fruit_mask & (1 << index):
                self.bonus_fruit_mask &= ~(1 << index)
                reward += self.BONUS_FRUIT_REWARD
        return reward

    def _food_progress_reward(self, previous_food_distance: int | None) -> float:
        current_food_distance = self._nearest_food_distance()
        if previous_food_distance is None or current_food_distance is None:
            return 0.0
        if current_food_distance < previous_food_distance:
            return self.FOOD_DISTANCE_GAIN_REWARD
        if current_food_distance > previous_food_distance:
            return self.FOOD_DISTANCE_LOSS_PENALTY
        return 0.0

    def _survival_reward(self, previous_ghost_distance: int) -> float:
        current_ghost_distance = self._nearest_ghost_distance()
        reward = self.SURVIVAL_REWARD
        if current_ghost_distance > previous_ghost_distance:
            reward += self.GHOST_DISTANCE_GAIN_REWARD
        elif (
            current_ghost_distance < previous_ghost_distance
            and current_ghost_distance <= self.GHOST_DANGER_DISTANCE
        ):
            reward += self.GHOST_DISTANCE_LOSS_PENALTY
        return reward

    def _nearest_ghost_distance(self) -> int:
        """Maze distance to the nearest ghost (uses precomputed BFS matrix)."""
        ghost_positions = [
            ghost_pos
            for index, ghost_pos in enumerate(self.ghost_positions)
            if self._ghost_can_catch(index)
        ]
        if not ghost_positions:
            return (self.height - 1) + (self.width - 1)
        return min(
            self._maze_distance(self.pacman_pos, ghost_pos)
            for ghost_pos in ghost_positions
        )

    def _nearest_food_distance(self) -> int | None:
        remaining_food = self._remaining_food_positions(self.food_mask)
        if not remaining_food:
            return None
        return min(self._maze_distance(self.pacman_pos, food_pos) for food_pos in remaining_food)

    def _remaining_food_positions(self, food_mask: int) -> list[Position]:
        return [
            food_pos
            for index, food_pos in enumerate(self.food_positions)
            if food_mask & (1 << index)
        ]

    def _auxiliary_state_features(
        self,
        pacman_pos: Position,
        ghost_positions: list[Position],
        food_mask: int,
        frightened_timer: int,
    ) -> list[float]:
        nearest_food = self._nearest_position(pacman_pos, self._remaining_food_positions(food_mask))
        nearest_ghost = self._nearest_position(pacman_pos, ghost_positions)
        features: list[float] = []
        features.extend(self._relative_position_features(pacman_pos, nearest_food))
        features.extend(self._relative_position_features(pacman_pos, nearest_ghost))
        features.extend(1.0 if self._move(pacman_pos, action) != pacman_pos else 0.0 for action in self.ACTIONS)
        features.extend(
            1.0
            if self._action_moves_toward_danger(pacman_pos, action, ghost_positions, frightened_timer)
            else 0.0
            for action in self.ACTIONS
        )
        return features

    def _nearest_position(self, origin: Position, positions: list[Position]) -> Position | None:
        if not positions:
            return None
        return min(positions, key=lambda position: self._maze_distance(origin, position))

    def _relative_position_features(self, origin: Position, target: Position | None) -> list[float]:
        if target is None:
            return [0.0, 0.0, 1.0]
        row_delta = (target[0] - origin[0]) / (self.height - 1)
        col_delta = (target[1] - origin[1]) / (self.width - 1)
        distance = self._manhattan(origin, target) / ((self.height - 1) + (self.width - 1))
        return [row_delta, col_delta, distance]

    def _action_moves_toward_danger(
        self,
        pacman_pos: Position,
        action: Action,
        ghost_positions: list[Position],
        frightened_timer: int = 0,
    ) -> bool:
        if frightened_timer > 0:
            return False
        if not ghost_positions:
            return False
        candidate = self._move(pacman_pos, action)
        nearest_distance = min(
            self._manhattan(candidate, ghost_pos)
            for ghost_pos in ghost_positions
        )
        return nearest_distance <= self.ACTION_DANGER_DISTANCE

    def _handle_caught(
        self,
        reward: float,
        bonus_fruit_eaten: bool = False,
        power_pellet_eaten: bool = False,
    ) -> StepResult:
        self.lives_remaining -= 1
        if self.lives_remaining <= 0:
            return self._finish(reward, "caught", bonus_fruit_eaten, power_pellet_eaten)

        self._reset_actor_positions()
        if self.steps >= self.max_steps:
            return StepResult(
                self._state(),
                reward + self.TIMEOUT_PENALTY,
                True,
                self._step_info("timeout", bonus_fruit_eaten, power_pellet_eaten),
            )

        return StepResult(
            self._state(),
            reward,
            False,
            self._step_info("life_lost", bonus_fruit_eaten, power_pellet_eaten),
        )

    def _finish(
        self,
        reward: float,
        event: str,
        bonus_fruit_eaten: bool = False,
        power_pellet_eaten: bool = False,
        ghosts_eaten: int = 0,
    ) -> StepResult:
        return StepResult(
            self._state(),
            reward,
            True,
            self._step_info(event, bonus_fruit_eaten, power_pellet_eaten, ghosts_eaten),
        )

    def _step_info(
        self,
        event: str,
        bonus_fruit_eaten: bool = False,
        power_pellet_eaten: bool = False,
        ghosts_eaten: int = 0,
    ) -> dict[str, object]:
        return {
            "event": event,
            "steps": self.steps,
            "lives": self.lives_remaining,
            "bonus_fruit_eaten": bonus_fruit_eaten,
            "bonus_fruits_remaining": self.bonus_fruit_mask.bit_count(),
            "power_pellet_eaten": power_pellet_eaten,
            "frightened_timer": self.frightened_timer,
            "ghosts_eaten": ghosts_eaten,
            "ghosts_returning": sum(1 for returning in self.ghost_returning if returning),
            "ghosts_respawning": sum(1 for timer in self.ghost_respawn_timers if timer > 0),
        }

    def _is_caught(self) -> bool:
        return any(
            self.pacman_pos == ghost_pos and self._ghost_can_catch(index)
            for index, ghost_pos in enumerate(self.ghost_positions)
        )

    def _eat_ghosts_at_pacman(self) -> int:
        if self.frightened_timer <= 0:
            return 0
        eaten = 0
        for index, ghost_pos in enumerate(self.ghost_positions):
            if (
                ghost_pos == self.pacman_pos
                and not self.ghost_returning[index]
                and self.ghost_respawn_timers[index] == 0
            ):
                self.ghost_returning[index] = True
                self.ghost_respawn_timers[index] = 0
                if index < len(self.ghost_last_actions):
                    self.ghost_last_actions[index] = None
                eaten += 1
        return eaten

    def _ghost_can_catch(self, ghost_index: int) -> bool:
        return (
            self.frightened_timer <= 0
            and not self.ghost_returning[ghost_index]
            and self.ghost_respawn_timers[ghost_index] == 0
        )

    @staticmethod
    def _manhattan(left: Position, right: Position) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    def _precompute_distances(self) -> dict[Position, dict[Position, int]]:
        """Run BFS once from every open cell to build a full pairwise distance matrix.

        Time: O(|open| * (V+E)) at init, paid once.
        Query: O(1) during episode steps.
        Memory: O(|open|^2), about 1000 entries for a 15x15 map with ~130 open cells.
        """
        open_cells = self._open_positions()
        dist_matrix: dict[Position, dict[Position, int]] = {}
        for source in open_cells:
            distances: dict[Position, int] = {source: 0}
            queue: deque[Position] = deque([source])
            while queue:
                position = queue.popleft()
                for action in self.ACTIONS:
                    neighbour = self._move(position, action, allow_ghost_door=True)
                    if neighbour == position or neighbour in distances:
                        continue
                    distances[neighbour] = distances[position] + 1
                    queue.append(neighbour)
            dist_matrix[source] = distances
        return dist_matrix
