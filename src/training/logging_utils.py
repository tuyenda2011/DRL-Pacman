from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

MetricValue = float | int | str
MetricRow = dict[str, MetricValue]


class MetricsCsvLogger:
    def __init__(
        self,
        output: str | Path,
        fieldnames: list[str],
        existing_rows: Iterable[dict[str, object]] | None = None,
        append: bool = False,
    ) -> None:
        self.output_path = Path(output)
        self.fieldnames = fieldnames
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append and self.output_path.exists() else "w"
        self.file: TextIO = self.output_path.open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w" or self.output_path.stat().st_size == 0:
            self.writer.writeheader()
        if existing_rows:
            self.writer.writerows(existing_rows)
        self.file.flush()

    def write(self, row: MetricRow) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> MetricsCsvLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_metric_rows(output: str | Path, up_to_episode: int | None = None) -> list[MetricRow]:
    output_path = Path(output)
    if not output_path.exists():
        return []

    rows: list[MetricRow] = []
    with output_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            episode = int(row.get("episode") or 0)
            if up_to_episode is not None and episode > up_to_episode:
                continue
            rows.append({key: _parse_metric_value(value) for key, value in row.items()})
    return rows


def append_training_run_log(
    history_output: str | Path,
    *,
    algorithm: str,
    args: argparse.Namespace,
    status: str,
    start_episode: int,
    final_episode: int,
    final_metrics: MetricRow | None,
) -> None:
    history_path = Path(history_output)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "status": status,
        "resume": bool(getattr(args, "resume", False)),
        "episode_range": {
            "start": start_episode,
            "final": final_episode,
            "trained": max(0, final_episode - start_episode + 1),
            "requested": args.episodes,
        },
        "artifacts": {
            "metrics": str(Path(args.output)),
            "eval_metrics": str(Path(args.eval_output)) if getattr(args, "eval_output", None) else None,
            "model": str(Path(args.model_output)),
            "checkpoint": str(Path(args.checkpoint_path)),
        },
        "config": _serializable_args(args),
        "final_metrics": final_metrics or {},
    }
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = repr(value)
    return result


def _parse_metric_value(value: str | None) -> MetricValue:
    if value is None:
        return ""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def print_training_header(
    algorithm: str,
    args: argparse.Namespace,
    process_steps: Iterable[str],
) -> None:
    print(f"\n=== Train {algorithm} on Mini Pacman ===")
    print("Training process:")
    for index, step in enumerate(process_steps, start=1):
        print(f"  {index}. {step}")
    print("Parameters:")
    print(
        "  "
        f"episodes={args.episodes}, max_steps={args.max_steps}, "
        f"gamma={args.discount_factor}, epsilon={args.epsilon_start}->{args.epsilon_min}, "
        f"epsilon_decay={args.epsilon_decay}, seed={args.seed}"
    )
    if hasattr(args, "ghost_count"):
        chase_probability = getattr(args, "ghost_chase_probability", "default")
        food_count = getattr(args, "food_count", "default")
        map_size = getattr(args, "map_size", "fixed")
        layout = getattr(args, "layout", "medium")
        print(
            "  "
            f"layout={layout}, map={map_size}, ghost_count={args.ghost_count}, food={food_count}, maze=fixed, "
            f"ghost_chase_probability={chase_probability}"
        )
    print()


def should_log_episode(episode: int, total_episodes: int, log_interval: int, event: str) -> bool:
    if event == "interrupted":
        return True
    if log_interval <= 0:
        return False
    return episode == 1 or episode == total_episodes or episode % log_interval == 0 or event == "win"


def print_episode_log(
    algorithm: str,
    episode: int,
    total_episodes: int,
    reward: float,
    steps: int,
    epsilon: float,
    event: str,
    loss: float | None = None,
    extra: str = "",
) -> None:
    loss_text = "" if loss is None else f" | loss={loss:.6f}"
    extra_text = "" if not extra else f" | {extra}"
    print(
        f"[{algorithm}] "
        f"episode {episode:>4}/{total_episodes:<4} | "
        f"reward={reward:>8.2f} | steps={steps:>3} | "
        f"epsilon={epsilon:.4f}{loss_text} | event={event}{extra_text}"
    )


def maybe_render_env(render_text: str, episode: int, render_interval: int) -> None:
    if render_interval > 0 and episode % render_interval == 0:
        print("Current map:")
        print(render_text)
        print()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string (e.g. '1h 23m 45s')."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_eta(elapsed_sec: float, episode: int, total_episodes: int) -> str:
    """Estimate remaining training time based on average time-per-episode so far."""
    if episode <= 0:
        return "?"
    avg_sec_per_episode = elapsed_sec / episode
    remaining_episodes = total_episodes - episode
    eta_sec = avg_sec_per_episode * remaining_episodes
    return format_duration(eta_sec)


def print_saved_outputs(
    output: str,
    model_output: str | None,
    history_output: str | None = None,
    status: str = "completed",
    elapsed_sec: float | None = None,
) -> None:
    if status == "interrupted":
        print("\nTraining interrupted.")
    elif status == "skipped":
        print("\nTraining skipped.")
    else:
        print("\nTraining completed.")
    if elapsed_sec is not None:
        print(f"Total time:       {format_duration(elapsed_sec)}")
    print(f"Metrics saved to: {Path(output)}")
    if model_output is None:
        print("Model saved to:   not saved")
    else:
        print(f"Model saved to:   {Path(model_output)}")
    if history_output:
        print(f"Run log saved to: {Path(history_output)}")
