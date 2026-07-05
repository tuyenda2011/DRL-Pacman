from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image, ImageDraw, ImageFont

from src.pacman_env.grid_world import MiniPacmanEnv, State


ACTION_ANGLES = {
    0: 90,
    1: 0,
    2: 270,
    3: 180,
}

BLACK = "#000000"
MAZE_BLUE = "#0033ff"
PACMAN_YELLOW = "#ffff3d"
FOOD_COLOR = "#f4b08a"
POWER_COLOR = "#ffe1c5"
GHOST_COLORS = ["#ff0000", "#ffb8ff", "#00ffff"]
FRIGHTENED_BLUE = "#1f51ff"
TEXT_COLOR = "#f4f4ff"
DOOR_COLOR = "#ffb8d8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a trained Mini Pacman model to an animated GIF.")
    parser.add_argument("--algorithm", choices=["dqn", "double_dqn"], default="dqn")
    parser.add_argument("--config", default="configs/dqn/dqn_lr_0005.yaml")
    parser.add_argument("--model-path", default="models/final/dqn/dqn_lr_0005.pt")
    parser.add_argument("--output", default="docs/assets/demo_dqn_lr_0005.gif")
    parser.add_argument("--seed", type=int, default=10046)
    parser.add_argument("--cell-size", type=int, default=24)
    parser.add_argument("--duration-ms", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--hold-final-frames", type=int, default=18)
    return parser


def load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Config file must contain a mapping: {path}")
    return config


def build_action_fn(
    algorithm: str,
    env: MiniPacmanEnv,
    model_path: str | Path,
    hidden_size: int,
    seed: int,
) -> Callable[[State], int]:
    if algorithm == "dqn":
        from src.algorithms.dqn.agent import DQNAgent

        agent = DQNAgent(
            state_size=env.vector_size,
            action_size=env.action_size,
            hidden_size=hidden_size,
            epsilon=0.0,
            seed=seed,
        )
    else:
        from src.algorithms.double_dqn.agent import DoubleDQNAgent

        agent = DoubleDQNAgent(
            state_size=env.vector_size,
            action_size=env.action_size,
            hidden_size=hidden_size,
            epsilon=0.0,
            seed=seed,
        )
    agent.load(model_path)
    return lambda state: agent.select_action(env.state_vector(state))


def render_frame(
    env: MiniPacmanEnv,
    *,
    algorithm_label: str,
    episode_reward: float,
    last_action: int,
    last_event: str,
    cell_size: int,
) -> Image.Image:
    margin = cell_size
    top_hud = int(cell_size * 2.7)
    bottom_hud = int(cell_size * 1.4)
    width = env.width * cell_size + margin * 2
    height = env.height * cell_size + top_hud + bottom_hud
    image = Image.new("RGB", (width, height), BLACK)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    score = max(0, int(round(episode_reward * 100)))
    title = f"{algorithm_label}  step {env.steps:03d}  score {score:05d}  {last_event}"
    draw.text((margin, int(cell_size * 0.55)), title, fill=TEXT_COLOR, font=font)
    draw.text((margin, int(cell_size * 1.45)), f"lives {env.lives_remaining}", fill=TEXT_COLOR, font=font)

    board_x = margin
    board_y = top_hud

    for row, col in env.walls:
        x1, y1 = board_x + col * cell_size, board_y + row * cell_size
        draw.rectangle((x1, y1, x1 + cell_size, y1 + cell_size), fill=BLACK)
        pad = max(2, int(cell_size * 0.12))
        draw.rectangle(
            (x1 + pad, y1 + pad, x1 + cell_size - pad, y1 + cell_size - pad),
            outline=MAZE_BLUE,
            width=max(2, int(cell_size * 0.10)),
        )

    for row, col in env.ghost_doors:
        cx, cy = cell_center(board_x, board_y, cell_size, row, col)
        draw.line((cx - cell_size * 0.38, cy, cx + cell_size * 0.38, cy), fill=DOOR_COLOR, width=2)

    power_pellets = set(env.power_pellet_positions)
    for index, (row, col) in enumerate(env.food_positions):
        if not env.food_mask & (1 << index):
            continue
        cx, cy = cell_center(board_x, board_y, cell_size, row, col)
        radius = cell_size * (0.22 if (row, col) in power_pellets else 0.08)
        color = POWER_COLOR if (row, col) in power_pellets else FOOD_COLOR
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)

    for index, (row, col) in enumerate(env.bonus_fruit_positions):
        if env.bonus_fruit_mask & (1 << index):
            cx, cy = cell_center(board_x, board_y, cell_size, row, col)
            draw.ellipse((cx - 4, cy, cx + 4, cy + 8), fill="#ff1b1c")
            draw.ellipse((cx + 2, cy, cx + 10, cy + 8), fill="#ff1b1c")
            draw.line((cx + 1, cy, cx + 5, cy - 8), fill="#32d96b", width=2)

    for ghost_index, (row, col) in enumerate(env.ghost_positions):
        draw_ghost(draw, board_x, board_y, cell_size, env, row, col, ghost_index)

    draw_pacman(draw, board_x, board_y, cell_size, env.pacman_pos[0], env.pacman_pos[1], last_action)
    return image


def cell_center(board_x: int, board_y: int, cell_size: int, row: int, col: int) -> tuple[float, float]:
    return board_x + (col + 0.5) * cell_size, board_y + (row + 0.5) * cell_size


def draw_pacman(
    draw: ImageDraw.ImageDraw,
    board_x: int,
    board_y: int,
    cell_size: int,
    row: int,
    col: int,
    action: int,
) -> None:
    cx, cy = cell_center(board_x, board_y, cell_size, row, col)
    radius = cell_size * 0.36
    start = ACTION_ANGLES.get(action, 0) + 28
    end = ACTION_ANGLES.get(action, 0) - 28
    draw.pieslice((cx - radius, cy - radius, cx + radius, cy + radius), start=end, end=start + 360, fill=PACMAN_YELLOW)
    draw.pieslice((cx - radius, cy - radius, cx + radius, cy + radius), start=end, end=start, fill=BLACK)


def draw_ghost(
    draw: ImageDraw.ImageDraw,
    board_x: int,
    board_y: int,
    cell_size: int,
    env: MiniPacmanEnv,
    row: int,
    col: int,
    ghost_index: int,
) -> None:
    cx, cy = cell_center(board_x, board_y, cell_size, row, col)
    color = GHOST_COLORS[ghost_index % len(GHOST_COLORS)]
    if env.frightened_timer > 0 and not env.ghost_respawn_timers[ghost_index]:
        color = FRIGHTENED_BLUE
    if env.ghost_returning[ghost_index]:
        draw_eyes(draw, cx, cy, cell_size)
        return

    width = cell_size * 0.72
    height = cell_size * 0.82
    left = cx - width / 2
    top = cy - height / 2
    right = cx + width / 2
    bottom = cy + height / 2
    draw.rounded_rectangle((left, top, right, bottom), radius=int(cell_size * 0.25), fill=color)
    foot_y = bottom - cell_size * 0.10
    for offset in (-0.24, 0.0, 0.24):
        fx = cx + offset * cell_size
        draw.polygon(
            [(fx - cell_size * 0.14, foot_y), (fx, bottom + cell_size * 0.10), (fx + cell_size * 0.14, foot_y)],
            fill=BLACK,
        )
    draw_eyes(draw, cx, cy - cell_size * 0.08, cell_size)


def draw_eyes(draw: ImageDraw.ImageDraw, cx: float, cy: float, cell_size: int) -> None:
    for dx in (-cell_size * 0.14, cell_size * 0.14):
        ex = cx + dx
        draw.ellipse((ex - 4, cy - 5, ex + 4, cy + 5), fill="#ffffff")
        draw.ellipse((ex - 1, cy - 2, ex + 3, cy + 2), fill="#1d4ed8")


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    env = MiniPacmanEnv(
        layout_name=str(config.get("layout", "medium")),
        max_steps=int(config.get("max_steps", 700)),
        max_lives=int(config.get("max_lives", 3)),
        ghost_count=int(config.get("ghost_count", 3)),
        ghost_chase_probability=float(config.get("ghost_chase_probability", 1.0)),
        seed=args.seed,
    )
    action_fn = build_action_fn(
        algorithm=args.algorithm,
        env=env,
        model_path=args.model_path,
        hidden_size=int(config.get("hidden_size", 128)),
        seed=int(config.get("seed", 42)),
    )

    frames: list[Image.Image] = []
    state = env.reset()
    total_reward = 0.0
    last_action = 1
    last_event = "start"
    label = f"{args.algorithm.upper()} lr=0.0005"
    frames.append(
        render_frame(
            env,
            algorithm_label=label,
            episode_reward=total_reward,
            last_action=last_action,
            last_event=last_event,
            cell_size=args.cell_size,
        )
    )

    for _ in range(args.max_frames):
        last_action = action_fn(state)
        result = env.step(last_action)
        state = result.state
        total_reward += result.reward
        last_event = str(result.info["event"])
        frames.append(
            render_frame(
                env,
                algorithm_label=label,
                episode_reward=total_reward,
                last_action=last_action,
                last_event=last_event,
                cell_size=args.cell_size,
            )
        )
        if result.done:
            break

    frames.extend([frames[-1].copy() for _ in range(args.hold_final_frames)])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Saved demo GIF: {output_path}")
    print(f"frames={len(frames)} steps={env.steps} reward={total_reward:.2f} event={last_event}")


if __name__ == "__main__":
    main()
