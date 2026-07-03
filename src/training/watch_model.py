from __future__ import annotations

import argparse
import math
import tkinter as tk
from pathlib import Path
from typing import Callable

from src.algorithms.q_learning.agent import QLearningAgent
from src.pacman_env.grid_world import MiniPacmanEnv, State
from src.training.checkpointing import (
    latest_checkpoint_path,
    load_deep_q_checkpoint,
    load_q_learning_checkpoint,
)
from src.training.config_utils import load_config_defaults


ACTION_NAMES = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}

BLACK = "#000000"
MAZE_BLUE = "#0033ff"
MAZE_GLOW = "#001a80"
PACMAN_YELLOW = "#ffff3d"
FOOD_COLOR = "#f4b08a"
ARCADE_WHITE = "#f4f4ff"
READY_YELLOW = "#ffe600"
FRIGHTENED_BLUE = "#1f51ff"
FRIGHTENED_FLASH = "#f4f4ff"
GHOST_DOOR_COLOR = "#ffb8d8"
FOOD_SIZE = 0.1
CAPSULE_SIZE = 0.25
GHOST_SIZE = 0.65
GHOST_COLORS = [
    "#ff0000",  # Blinky
    "#ffb8ff",  # Pinky
    "#00ffff",  # Inky
]
MODEL_EXTENSIONS = {
    "q_learning": "pkl",
    "dqn": "pt",
    "double_dqn": "pt",
}

CONFIG_PATHS = {
    "q_learning": "configs/q_learning/q_learning_lr_001.yaml",
    "dqn": "configs/dqn/dqn_lr_001.yaml",
    "double_dqn": "configs/double_dqn/double_dqn_lr_001.yaml",
}

WATCH_CONFIG_KEYS = {
    "algorithm",
    "ghost_count",
    "ghost_chase_probability",
    "hidden_size",
    "layout",
    "max_lives",
    "max_steps",
    "model_output",
    "checkpoint_path",
    "run_name",
    "seed",
}


class PacmanViewer:
    TOP_HUD_CELLS = 4.8
    BOTTOM_HUD_CELLS = 1.5
    SIDE_MARGIN_CELLS = 0.85
    CANVAS_PADDING = 12
    WINDOW_HEADROOM = 96

    def __init__(
        self,
        root: tk.Tk,
        env: MiniPacmanEnv,
        action_fn: Callable[[State], int],
        algorithm: str,
        delay_ms: int,
        episodes: int,
        cell_size: int,
        map_only: bool = False,
        ghost_demo: bool = False,
    ) -> None:
        self.root = root
        self.env = env
        self.action_fn = action_fn
        self.algorithm = algorithm
        self.delay_ms = delay_ms
        self.episodes = episodes
        self.map_only = map_only
        self.ghost_demo = ghost_demo
        self.cell_size = self._fit_cell_size(cell_size)

        self.episode = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.state: State | None = None
        self.done = False
        self.last_action = "-"
        self.last_action_value: int | None = None
        self.last_event = "ready"
        self.animation_frame = 0
        self.best_score = 0
        self.top_hud_height = int(self.cell_size * self.TOP_HUD_CELLS)
        self.bottom_hud_height = int(self.cell_size * self.BOTTOM_HUD_CELLS)
        self.board_x = int(self.cell_size * self.SIDE_MARGIN_CELLS)
        self.board_y = self.top_hud_height
        self.board_width = env.width * self.cell_size
        self.board_height = env.height * self.cell_size
        self.power_pellets = set(env.power_pellet_positions)

        self.root.title(f"Mini Pacman - {algorithm}")
        self.root.configure(bg=BLACK)
        self.canvas = tk.Canvas(
            root,
            width=self.board_width + (self.board_x * 2),
            height=self.top_hud_height + self.board_height + self.bottom_hud_height,
            bg=BLACK,
            highlightthickness=0,
        )
        self.canvas.pack(padx=self.CANVAS_PADDING, pady=self.CANVAS_PADDING)

        self.start_episode()

    def _fit_cell_size(self, requested_cell_size: int) -> int:
        screen_width, screen_height = self._available_screen_size()
        canvas_extra_width = self.SIDE_MARGIN_CELLS * 2
        canvas_extra_height = self.TOP_HUD_CELLS + self.BOTTOM_HUD_CELLS
        max_by_width = int(
            (screen_width - (self.CANVAS_PADDING * 2))
            / (self.env.width + canvas_extra_width)
        )
        max_by_height = int(
            (screen_height - self.WINDOW_HEADROOM - (self.CANVAS_PADDING * 2))
            / (self.env.height + canvas_extra_height)
        )
        fitted = min(requested_cell_size, max_by_width, max_by_height)
        return max(14, fitted)

    def _available_screen_size(self) -> tuple[int, int]:
        dpi_scale = max(float(self.root.tk.call("tk", "scaling")) / (96 / 72), 1.0)
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            success = ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            if success:
                return (
                    int((rect.right - rect.left) / dpi_scale),
                    int((rect.bottom - rect.top) / dpi_scale),
                )
        except (AttributeError, ImportError, OSError):
            pass

        return (
            int(self.root.winfo_screenwidth() / dpi_scale),
            int(self.root.winfo_screenheight() / dpi_scale),
        )

    def start_episode(self) -> None:
        self.episode += 1
        self.step_count = 0
        self.total_reward = 0.0
        self.state = self.env.reset()
        self.done = False
        self.last_action = "-"
        self.last_action_value = None
        self.animation_frame = 0
        if self.ghost_demo:
            self.last_event = "ghost demo"
        else:
            self.last_event = "map" if self.map_only else "start"
        self.draw()
        if self.map_only:
            return
        self.root.after(self.delay_ms, self.step)

    def step(self) -> None:
        if self.done:
            if self.episode < self.episodes:
                self.root.after(max(self.delay_ms * 3, 500), self.start_episode)
            return

        if self.state is None:
            return

        if self.ghost_demo:
            self._step_ghost_demo()
            self.draw()
            self.root.after(self.delay_ms, self.step)
            return

        action = self.action_fn(self.state)
        result = self.env.step(action)
        self.animation_frame += 1
        self.state = result.state
        self.step_count += 1
        self.total_reward += result.reward
        self.done = result.done
        self.last_action = ACTION_NAMES[action]
        self.last_action_value = action
        if result.info.get("ghosts_eaten"):
            self.last_event = "ghost eaten"
        elif result.info.get("power_pellet_eaten"):
            self.last_event = "power"
        elif result.info.get("bonus_fruit_eaten"):
            self.last_event = "fruit"
        else:
            self.last_event = str(result.info["event"])

        self.draw()
        self.root.after(self.delay_ms, self.step)

    def _step_ghost_demo(self) -> None:
        self.env.steps += 1
        self.env.ghost_positions = self.env._move_ghosts()
        self.state = self.env._state()
        self.animation_frame += 1
        self.step_count += 1
        self.last_action = "GHOST"
        self.last_action_value = None
        self.last_event = self.env._ghost_mode()
        self.done = self.env.steps >= self.env.max_steps

    def draw(self) -> None:
        self.canvas.delete("all")
        self.draw_title_hud()
        self.draw_maze_background()
        self.draw_maze_walls()
        self.draw_ghost_doors()

        for index, (row, col) in enumerate(self.env.food_positions):
            if self.env.food_mask & (1 << index):
                self.draw_food(row, col)

        for index, (row, col) in enumerate(self.env.bonus_fruit_positions):
            if self.env.bonus_fruit_mask & (1 << index):
                self.draw_cherry_pair_at_cell(row, col)

        for ghost_index, (ghost_row, ghost_col) in enumerate(self.env.ghost_positions):
            self.draw_ghost(ghost_row, ghost_col, ghost_index)
        if not self.ghost_demo:
            pacman_row, pacman_col = self.env.pacman_pos
            self.draw_pacman(pacman_row, pacman_col)
        self.draw_ready_banner()
        self.draw_bottom_hud()

    def draw_title_hud(self) -> None:
        width = int(self.canvas["width"])
        score = max(0, int(round(self.total_reward * 100)))
        self.best_score = max(self.best_score, score)
        label_font = ("Courier New", max(8, int(self.cell_size * 0.25)), "bold")
        score_font = ("Courier New", max(9, int(self.cell_size * 0.28)), "bold")
        status_font = ("Courier New", max(8, int(self.cell_size * 0.23)), "bold")

        self.canvas.create_text(
            width * 0.20,
            self.cell_size * 0.85,
            text="1UP",
            fill=ARCADE_WHITE,
            font=label_font,
        )
        self.canvas.create_text(
            width * 0.54,
            self.cell_size * 0.85,
            text="HIGH SCORE",
            fill=ARCADE_WHITE,
            font=label_font,
        )
        self.canvas.create_text(
            width * 0.20,
            self.cell_size * 1.20,
            text=f"{score:05d}",
            fill=ARCADE_WHITE,
            font=score_font,
        )
        self.canvas.create_text(
            width * 0.54,
            self.cell_size * 1.20,
            text=f"{self.best_score:05d}",
            fill=ARCADE_WHITE,
            font=score_font,
        )
        self.canvas.create_text(
            width / 2,
            self.cell_size * 2.65,
            text=(
                f"{self.algorithm.upper()}  "
                f"EP {self.episode}/{self.episodes}  "
                f"STEP {self.step_count}  {self.last_action}  {self.last_event.upper()}"
            ),
            fill=ARCADE_WHITE,
            font=status_font,
        )

    def draw_maze_background(self) -> None:
        self.canvas.create_rectangle(
            self.board_x,
            self.board_y,
            self.board_x + self.board_width,
            self.board_y + self.board_height,
            fill=BLACK,
            outline="",
        )

    def draw_maze_walls(self) -> None:
        wall_width = max(3, int(self.cell_size * 0.13))
        glow_width = wall_width + 2
        for row, col in self.env.walls:
            x1, y1 = self.cell_origin(row, col)
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            neighbors = {
                "top": (row - 1, col),
                "right": (row, col + 1),
                "bottom": (row + 1, col),
                "left": (row, col - 1),
            }
            edges = []
            if self._is_open_or_outside(neighbors["top"]):
                edges.append((x1, y1, x2, y1))
            if self._is_open_or_outside(neighbors["right"]):
                edges.append((x2, y1, x2, y2))
            if self._is_open_or_outside(neighbors["bottom"]):
                edges.append((x1, y2, x2, y2))
            if self._is_open_or_outside(neighbors["left"]):
                edges.append((x1, y1, x1, y2))
            for edge in edges:
                self.canvas.create_line(*edge, fill=MAZE_GLOW, width=glow_width, capstyle=tk.ROUND)
                self.canvas.create_line(*edge, fill=MAZE_BLUE, width=wall_width, capstyle=tk.ROUND)

    def draw_ghost_doors(self) -> None:
        door_width = max(2, int(self.cell_size * 0.10))
        rows: dict[int, list[int]] = {}
        for row, col in self.env.ghost_doors:
            rows.setdefault(row, []).append(col)
        for row, cols in rows.items():
            sorted_cols = sorted(cols)
            start_col = sorted_cols[0]
            previous_col = sorted_cols[0]
            for col in sorted_cols[1:] + [None]:
                if col is not None and col == previous_col + 1:
                    previous_col = col
                    continue
                x1, y1 = self.cell_origin(row, start_col)
                x2, _ = self.cell_origin(row, previous_col)
                y = y1 + self.cell_size * 0.54
                self.canvas.create_line(
                    x1 + self.cell_size * 0.08,
                    y,
                    x2 + self.cell_size * 0.92,
                    y,
                    fill=GHOST_DOOR_COLOR,
                    width=door_width,
                    capstyle=tk.BUTT,
                )
                if col is not None:
                    start_col = col
                    previous_col = col

    def draw_food(self, row: int, col: int) -> None:
        center_x, center_y = self.cell_center(row, col)
        is_power_pellet = (row, col) in self.power_pellets
        radius = self.cell_size * (CAPSULE_SIZE if is_power_pellet else FOOD_SIZE)
        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=FOOD_COLOR,
            outline="",
        )
        if is_power_pellet:
            self.canvas.create_oval(
                center_x - radius * 0.62,
                center_y - radius * 0.62,
                center_x + radius * 0.62,
                center_y + radius * 0.62,
                fill="#ffe1c5",
                outline="",
            )

    def draw_pacman(self, row: int, col: int) -> None:
        pad = self.cell_size * 0.14
        cell_x, cell_y = self.cell_origin(row, col)
        x1 = cell_x + pad
        y1 = cell_y + pad
        x2 = cell_x + self.cell_size - pad
        y2 = cell_y + self.cell_size - pad
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        radius = (x2 - x1) / 2
        direction = self.last_action_value if self.last_action_value is not None else 1
        mouth_angle = 38 if self.animation_frame % 2 == 0 else 18
        direction_angles = {
            0: -90,
            1: 0,
            2: 90,
            3: 180,
        }
        angle = math.radians(direction_angles[direction])
        upper = angle + math.radians(mouth_angle)
        lower = angle - math.radians(mouth_angle)

        self.canvas.create_oval(
            x1,
            y1,
            x2,
            y2,
            fill=PACMAN_YELLOW,
            outline="#fff38a",
            width=max(1, int(self.cell_size * 0.05)),
        )
        self.canvas.create_polygon(
            center_x,
            center_y,
            center_x + math.cos(upper) * radius * 1.08,
            center_y + math.sin(upper) * radius * 1.08,
            center_x + math.cos(lower) * radius * 1.08,
            center_y + math.sin(lower) * radius * 1.08,
            fill="#171717",
            outline="#171717",
        )

    def draw_ghost(self, row: int, col: int, ghost_index: int) -> None:
        is_returning = self.env.ghost_returning[ghost_index]
        is_respawning = self.env.ghost_respawn_timers[ghost_index] > 0
        color = GHOST_COLORS[ghost_index % len(GHOST_COLORS)]
        if self.env.frightened_timer > 0 and not is_respawning:
            color = FRIGHTENED_BLUE
            if self.env.frightened_timer <= 8 and self.animation_frame % 2 == 0:
                color = FRIGHTENED_FLASH
        center_x, center_y = self.cell_center(row, col)
        center_y += math.sin((self.animation_frame + ghost_index) * math.pi / 2) * self.cell_size * 0.035
        width = self.cell_size * 0.78
        height = self.cell_size * 0.88
        left = center_x - width / 2
        right = center_x + width / 2
        top = center_y - height * 0.48
        bottom = center_y + height * 0.46
        radius = width / 2
        side_top = top + radius
        foot_top = bottom - height * 0.18

        if is_returning:
            self.draw_ghost_eyes(
                center_x,
                top + height * 0.40,
                width,
                height,
                row,
                col,
                self.env.ghost_starts[ghost_index],
            )
            return

        body_points: list[float] = [left, foot_top, left, side_top]
        for step in range(18):
            theta = math.pi - (math.pi * step / 17)
            body_points.extend(
                [
                    center_x + math.cos(theta) * radius,
                    top + radius - math.sin(theta) * radius,
                ]
            )
        body_points.extend([right, foot_top])

        scallop_count = 4
        samples_per_scallop = 8
        for sample in range(scallop_count * samples_per_scallop + 1):
            progress = sample / (scallop_count * samples_per_scallop)
            x = right - progress * width
            wave = abs(math.sin(progress * scallop_count * math.pi + self.animation_frame * 0.9))
            y = foot_top + wave * (bottom - foot_top)
            body_points.extend([x, y])

        self.canvas.create_polygon(
            body_points,
            fill=color,
            outline="#050505",
            width=max(1, int(self.cell_size * 0.035)),
        )

        self.draw_ghost_eyes(center_x, top + height * 0.40, width, height, row, col, self.env.pacman_pos)

    def draw_ghost_eyes(
        self,
        center_x: float,
        eye_y: float,
        width: float,
        height: float,
        row: int,
        col: int,
        target: tuple[int, int],
    ) -> None:
        eye_radius_x = width * 0.13
        eye_radius_y = height * 0.17
        pupil_radius = width * 0.055
        eye_offset_x = width * 0.20
        look_x = _sign(target[1] - col) * eye_radius_x * 0.34
        look_y = _sign(target[0] - row) * eye_radius_y * 0.24
        for eye_x in (center_x - eye_offset_x, center_x + eye_offset_x):
            self.canvas.create_oval(
                eye_x - eye_radius_x,
                eye_y - eye_radius_y,
                eye_x + eye_radius_x,
                eye_y + eye_radius_y,
                fill="#ffffff",
                outline="",
            )
            self.canvas.create_oval(
                eye_x + look_x - pupil_radius,
                eye_y + look_y - pupil_radius,
                eye_x + look_x + pupil_radius,
                eye_y + look_y + pupil_radius,
                fill="#1d4ed8",
                outline="",
            )

    def draw_ready_banner(self) -> None:
        if self.map_only or self.ghost_demo or self.step_count > 0:
            return
        font = ("Courier New", max(13, int(self.cell_size * 0.44)), "bold")
        x = self.board_x + self.board_width / 2
        y = self.board_y + self.board_height * 0.58
        padding_x = self.cell_size * 1.45
        padding_y = self.cell_size * 0.40
        self.canvas.create_rectangle(
            x - padding_x,
            y - padding_y,
            x + padding_x,
            y + padding_y,
            fill=BLACK,
            outline="",
        )
        self.canvas.create_text(
            x,
            y,
            text="READY!",
            fill=READY_YELLOW,
            font=font,
        )

    def draw_bottom_hud(self) -> None:
        y = self.board_y + self.board_height + self.cell_size * 0.68
        x = self.board_x + self.board_width - self.cell_size * 4.9
        fruit_x = x
        if not self.ghost_demo:
            for index in range(self.env.lives_remaining):
                self.draw_life_icon(x + index * self.cell_size * 0.48, y)
            fruit_x = x + self.cell_size * 1.8
        eaten_bonus_fruits = len(self.env.bonus_fruit_positions) - self.env.bonus_fruit_mask.bit_count()
        for index in range(eaten_bonus_fruits):
            self.draw_cherry_pair(fruit_x + index * self.cell_size * 0.48, y, self.cell_size * 0.62)

    def draw_life_icon(self, x: float, y: float) -> None:
        radius = self.cell_size * 0.16
        self.canvas.create_arc(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            start=35,
            extent=290,
            fill=PACMAN_YELLOW,
            outline=PACMAN_YELLOW,
        )

    def draw_cherry_pair_at_cell(self, row: int, col: int) -> None:
        center_x, center_y = self.cell_center(row, col)
        self.draw_cherry_pair(center_x, center_y + self.cell_size * 0.08, self.cell_size)

    def draw_cherry_pair(self, x: float, y: float, scale: float) -> None:
        radius = scale * 0.13
        left_x = x - scale * 0.11
        right_x = x + scale * 0.11
        cherry_y = y + scale * 0.08
        stem_top_x = x + scale * 0.05
        stem_top_y = y - scale * 0.28
        stem_color = "#32d96b"
        self.canvas.create_oval(
            left_x - radius,
            cherry_y - radius,
            left_x + radius,
            cherry_y + radius,
            fill="#ff1b1c",
            outline="#8b0000",
            width=max(1, int(scale * 0.035)),
        )
        self.canvas.create_oval(
            right_x - radius,
            cherry_y - radius,
            right_x + radius,
            cherry_y + radius,
            fill="#ff1b1c",
            outline="#8b0000",
            width=max(1, int(scale * 0.035)),
        )
        self.canvas.create_line(
            left_x,
            cherry_y - radius * 0.65,
            stem_top_x,
            stem_top_y,
            fill=stem_color,
            width=max(1, int(scale * 0.045)),
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.canvas.create_line(
            right_x,
            cherry_y - radius * 0.65,
            stem_top_x,
            stem_top_y,
            fill=stem_color,
            width=max(1, int(scale * 0.045)),
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.canvas.create_oval(
            stem_top_x,
            stem_top_y - scale * 0.06,
            stem_top_x + scale * 0.17,
            stem_top_y + scale * 0.04,
            fill="#69e36f",
            outline="#1f8f3a",
        )

    def cell_origin(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.board_x + col * self.cell_size,
            self.board_y + row * self.cell_size,
        )

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        x, y = self.cell_origin(row, col)
        return x + self.cell_size / 2, y + self.cell_size / 2

    def _is_open_or_outside(self, position: tuple[int, int]) -> bool:
        row, col = position
        if row < 0 or row >= self.env.height or col < 0 or col >= self.env.width:
            return True
        return position not in self.env.walls

    def _select_power_pellets(self) -> set[tuple[int, int]]:
        corners = [
            (0, 0),
            (0, self.env.width - 1),
            (self.env.height - 1, 0),
            (self.env.height - 1, self.env.width - 1),
        ]
        selected: set[tuple[int, int]] = set()
        for corner in corners:
            if not self.env.food_positions:
                break
            selected.add(
                min(
                    self.env.food_positions,
                    key=lambda pos: abs(pos[0] - corner[0]) + abs(pos[1] - corner[1]),
                )
            )
        return selected


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def build_action_fn(
    args: argparse.Namespace,
    env: MiniPacmanEnv,
    model_path: Path,
    model_source: str,
) -> Callable[[State], int]:
    if args.algorithm == "q_learning":
        agent = QLearningAgent(action_size=env.action_size, epsilon=0.0, seed=args.seed)
        if model_source == "checkpoint":
            load_q_learning_checkpoint(agent, model_path)
            agent.epsilon = 0.0
        else:
            agent.load(model_path)
        return lambda state: agent.select_action(state)

    if args.algorithm == "dqn":
        try:
            from src.algorithms.dqn.agent import DQNAgent
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise SystemExit("PyTorch is not installed. Install dependencies first: pip install -r requirements.txt")

        agent = DQNAgent(
            state_size=env.vector_size,
            action_size=env.action_size,
            hidden_size=args.hidden_size,
            epsilon=0.0,
            seed=args.seed,
        )
        if model_source == "checkpoint":
            load_deep_q_checkpoint(agent, model_path)
            agent.epsilon = 0.0
        else:
            agent.load(model_path)
        return lambda state: agent.select_action(env.state_vector(state))

    if args.algorithm == "double_dqn":
        try:
            from src.algorithms.double_dqn.agent import DoubleDQNAgent
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise SystemExit("PyTorch is not installed. Install dependencies first: pip install -r requirements.txt")

        agent = DoubleDQNAgent(
            state_size=env.vector_size,
            action_size=env.action_size,
            hidden_size=args.hidden_size,
            epsilon=0.0,
            seed=args.seed,
        )
        if model_source == "checkpoint":
            load_deep_q_checkpoint(agent, model_path)
            agent.epsilon = 0.0
        else:
            agent.load(model_path)
        return lambda state: agent.select_action(env.state_vector(state))

    raise SystemExit(f"Unknown algorithm: {args.algorithm}")


def resolve_model_source(args: argparse.Namespace) -> tuple[Path, str]:
    if args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise SystemExit(f"Model file not found: {model_path}")
        source = "checkpoint" if "_checkpoint" in model_path.stem else "model"
        return model_path, source

    model_path = Path(args.model_output or _default_model_output(args.algorithm, args.run_name))
    checkpoint_base = Path(args.checkpoint_path or _default_checkpoint_path(args.algorithm, args.run_name))
    checkpoint_path = latest_checkpoint_path(checkpoint_base)

    if args.prefer_checkpoint and checkpoint_path.exists():
        return checkpoint_path, "checkpoint"
    if model_path.exists():
        return model_path, "model"
    if checkpoint_path.exists():
        return checkpoint_path, "checkpoint"

    raise SystemExit(
        "No trained model or checkpoint found.\n"
        f"Looked for final model: {model_path}\n"
        f"Looked for checkpoint:  {checkpoint_base}"
    )


def _default_model_output(algorithm: str, run_name: str) -> str:
    extension = MODEL_EXTENSIONS[algorithm]
    return f"models/final/{algorithm}/{run_name}.{extension}"


def _default_checkpoint_path(algorithm: str, run_name: str) -> str:
    return f"models/checkpoints/{algorithm}/{run_name}/{run_name}_checkpoint.pkl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch a trained Mini Pacman model in a GUI window.")
    parser.add_argument("--algorithm", choices=["q_learning", "dqn", "double_dqn"], default="q_learning")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--prefer-checkpoint", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--layout", choices=sorted(MiniPacmanEnv.LAYOUTS), default="medium")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-lives", type=int, default=3)
    parser.add_argument("--delay-ms", type=int, default=500)
    parser.add_argument("--cell-size", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ghost-count", type=int, choices=[2, 3], default=3)
    parser.add_argument("--ghost-chase-probability", type=float, default=1.0)
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--ghost-demo", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    config_probe, _ = parser.parse_known_args()
    config_path = config_probe.config or CONFIG_PATHS[config_probe.algorithm]
    parser.set_defaults(config=config_path)
    config_defaults = load_config_defaults(config_path, parser)
    parser.set_defaults(
        **{
            key: value
            for key, value in config_defaults.items()
            if key in WATCH_CONFIG_KEYS
        }
    )
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = Path(args.config).stem

    for key in ("model_output", "checkpoint_path"):
        value = getattr(args, key, None)
        if isinstance(value, str):
            setattr(args, key, value.format(run_name=args.run_name, algorithm=args.algorithm))

    return args


def main() -> None:
    args = parse_args()

    env = MiniPacmanEnv(
        layout_name=args.layout,
        max_steps=args.max_steps,
        max_lives=args.max_lives,
        ghost_count=args.ghost_count,
        ghost_chase_probability=args.ghost_chase_probability,
        seed=args.seed,
    )
    if args.map_only or args.ghost_demo:
        action_fn = lambda _state: 1
    else:
        model_path, model_source = resolve_model_source(args)
        print(f"Watching {model_source}: {model_path}")
        action_fn = build_action_fn(args, env, model_path, model_source)

    if args.ghost_demo:
        viewer_name = "ghost demo"
    else:
        viewer_name = "map" if args.map_only else f"{args.algorithm} {args.run_name}"

    root = tk.Tk()
    PacmanViewer(
        root=root,
        env=env,
        action_fn=action_fn,
        algorithm=viewer_name,
        delay_ms=args.delay_ms,
        episodes=args.episodes,
        cell_size=args.cell_size,
        map_only=args.map_only,
        ghost_demo=args.ghost_demo,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
