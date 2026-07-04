from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.pacman_env.grid_world import MiniPacmanEnv
from src.training.checkpointing import latest_checkpoint_path


MODULES = {
    "q_learning": "src.training.train_q_learning",
    "dqn": "src.training.train_dqn",
    "double_dqn": "src.training.train_double_dqn",
}

MAIN_CONFIGS = [
    Path("configs/q_learning/q_learning_lr_001.yaml"),
    Path("configs/dqn/dqn_lr_001.yaml"),
    Path("configs/double_dqn/double_dqn_lr_001.yaml"),
]

LR_SWEEP_CONFIGS = [
    Path("configs/q_learning/q_learning_lr_0001.yaml"),
    Path("configs/dqn/dqn_lr_0001.yaml"),
    Path("configs/double_dqn/double_dqn_lr_0001.yaml"),
    Path("configs/q_learning/q_learning_lr_0005.yaml"),
    Path("configs/dqn/dqn_lr_0005.yaml"),
    Path("configs/double_dqn/double_dqn_lr_0005.yaml"),
    Path("configs/q_learning/q_learning_lr_001.yaml"),
    Path("configs/dqn/dqn_lr_001.yaml"),
    Path("configs/double_dqn/double_dqn_lr_001.yaml"),
]

MODEL_EXTENSIONS = {
    "q_learning": "pkl",
    "dqn": "pt",
    "double_dqn": "pt",
}


@dataclass(frozen=True)
class RunConfig:
    path: Path
    algorithm: str
    run_name: str
    episodes: int
    layout: str
    max_steps: int
    max_lives: int
    ghost_count: int
    ghost_chase_probability: float
    hidden_size: int | None
    output: Path
    eval_output: Path
    history_output: Path
    model_output: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class RunStatus:
    state: str
    episode: int
    requested: int
    reason: str
    checkpoint_path: Path | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pacman training experiments from YAML configs.")
    parser.add_argument("--all-lr", action="store_true", help="Run all 9 learning-rate configs.")
    parser.add_argument("--resume", action="store_true", help="Resume every selected run from its latest checkpoint.")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume unfinished runs with compatible checkpoints; start new runs otherwise.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip runs that already reached the configured episode count.",
    )
    parser.add_argument(
        "--restart-incompatible",
        action="store_true",
        help="Delete artifacts for incompatible runs and train those configs from scratch.",
    )
    parser.add_argument("--status", action="store_true", help="Show run status and exit without training.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip compare_runs after training.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--plot-output", default="experiments/plots/training_comparison.png")
    parser.add_argument("configs", nargs="*", help="Optional explicit config paths to run.")
    args = parser.parse_args()

    configs = [Path(config) for config in args.configs]
    if not configs:
        configs = LR_SWEEP_CONFIGS if args.all_lr else MAIN_CONFIGS

    runs = [_load_run_config(config) for config in configs]
    statuses = [(run, _status_for_run(run)) for run in runs]
    if args.status:
        _print_statuses(statuses)
        return

    ran_training = False
    skip_completed = args.skip_completed or args.auto_resume
    for run, status in statuses:
        _print_status(run, status)
        if status.state == "completed" and skip_completed:
            print(f"  -> skip completed ({status.episode}/{status.requested})", flush=True)
            continue
        if status.state == "incompatible":
            if args.restart_incompatible:
                _remove_run_artifacts(run, dry_run=args.dry_run)
                status = RunStatus("new", 0, run.episodes, "restarting after incompatible artifacts")
            else:
                print(f"  -> skip incompatible checkpoint: {status.reason}", flush=True)
                continue

        module = _module_for_config(run.path)
        command = [sys.executable, "-m", module, "--config", str(run.path)]
        if args.resume or (args.auto_resume and status.state == "resumable"):
            command.append("--resume")
        elif args.auto_resume and status.state == "incomplete":
            print("  -> no compatible checkpoint found; starting this config from scratch", flush=True)
        _run(command, dry_run=args.dry_run)
        ran_training = True

    if not args.skip_plots and (ran_training or args.dry_run):
        _run(
            [
                sys.executable,
                "-m",
                "src.training.compare_runs",
                "--output",
                args.plot_output,
            ],
            dry_run=args.dry_run,
        )


def _load_run_config(config: Path) -> RunConfig:
    path = Path(config)
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {path}")

    algorithm = str(raw.get("algorithm") or _algorithm_from_path(path))
    run_name = str(raw.get("run_name") or path.stem)
    model_extension = MODEL_EXTENSIONS.get(algorithm, "pt")

    def value(name: str, default: Any) -> Any:
        item = raw.get(name, default)
        if isinstance(item, str):
            return item.format(run_name=run_name, algorithm=algorithm)
        return item

    return RunConfig(
        path=path,
        algorithm=algorithm,
        run_name=run_name,
        episodes=int(value("episodes", 0)),
        layout=str(value("layout", "medium")),
        max_steps=int(value("max_steps", 500)),
        max_lives=int(value("max_lives", 3)),
        ghost_count=int(value("ghost_count", 3)),
        ghost_chase_probability=float(value("ghost_chase_probability", 1.0)),
        hidden_size=int(value("hidden_size", 128)) if algorithm in {"dqn", "double_dqn"} else None,
        output=Path(value("output", "experiments/metrics/{algorithm}/{run_name}_metrics.csv")),
        eval_output=Path(value("eval_output", "experiments/metrics/{algorithm}/{run_name}_eval_metrics.csv")),
        history_output=Path(value("history_output", "experiments/history/{algorithm}/{run_name}_history.jsonl")),
        model_output=Path(value("model_output", f"models/final/{{algorithm}}/{{run_name}}.{model_extension}")),
        checkpoint_path=Path(
            value("checkpoint_path", "models/checkpoints/{algorithm}/{run_name}/{run_name}_checkpoint.pkl")
        ),
    )


def _status_for_run(run: RunConfig) -> RunStatus:
    requested = run.episodes
    checkpoint = latest_checkpoint_path(run.checkpoint_path)
    if checkpoint.exists():
        checkpoint_status = _status_from_checkpoint(run, checkpoint)
        if checkpoint_status.state == "incompatible":
            return checkpoint_status
        if checkpoint_status.episode >= requested:
            return RunStatus(
                "completed",
                checkpoint_status.episode,
                requested,
                f"checkpoint reached requested episodes: {checkpoint}",
                checkpoint,
            )
        return checkpoint_status

    history_status = _status_from_history(run)
    metric_episode = _last_metric_episode(run.output)
    best_episode = max(history_status.episode, metric_episode)
    if history_status.state == "completed" or (metric_episode >= requested and run.model_output.exists()):
        return RunStatus("completed", best_episode, requested, "completed history/metrics found")
    if best_episode > 0:
        return RunStatus("incomplete", best_episode, requested, "metrics/history found but no checkpoint to resume")
    return RunStatus("new", 0, requested, "no checkpoint or metrics found")


def _status_from_checkpoint(run: RunConfig, checkpoint: Path) -> RunStatus:
    try:
        with checkpoint.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError) as exc:
        return RunStatus("incompatible", 0, run.episodes, f"cannot read {checkpoint}: {exc}", checkpoint)

    episode = int(payload.get("episode", _checkpoint_episode(checkpoint)))
    metadata = payload.get("metadata") or {}
    problems = _checkpoint_compatibility_problems(run, metadata)
    if problems:
        return RunStatus("incompatible", episode, run.episodes, "; ".join(problems), checkpoint)
    return RunStatus("resumable", episode, run.episodes, f"compatible checkpoint: {checkpoint}", checkpoint)


def _checkpoint_compatibility_problems(run: RunConfig, metadata: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    expected = {
        "layout": run.layout,
        "ghost_count": run.ghost_count,
        "ghost_chase_probability": run.ghost_chase_probability,
        "max_steps": run.max_steps,
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if not _same_value(actual, expected_value):
            problems.append(f"{key} checkpoint={actual!r} config={expected_value!r}")

    env = MiniPacmanEnv(
        layout_name=run.layout,
        max_steps=run.max_steps,
        max_lives=run.max_lives,
        ghost_count=run.ghost_count,
        ghost_chase_probability=run.ghost_chase_probability,
    )
    expected_state_size = env.vector_size if run.algorithm in {"dqn", "double_dqn"} else len(env._state())
    if not _same_value(metadata.get("state_size"), expected_state_size):
        problems.append(f"state_size checkpoint={metadata.get('state_size')!r} config={expected_state_size!r}")

    if run.algorithm in {"dqn", "double_dqn"}:
        if not _same_value(metadata.get("hidden_size"), run.hidden_size):
            problems.append(f"hidden_size checkpoint={metadata.get('hidden_size')!r} config={run.hidden_size!r}")
    return problems


def _status_from_history(run: RunConfig) -> RunStatus:
    if not run.history_output.exists():
        return RunStatus("new", 0, run.episodes, "history not found")
    last_record: dict[str, Any] | None = None
    with run.history_output.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                last_record = json.loads(line)
            except json.JSONDecodeError:
                continue
    if not last_record:
        return RunStatus("new", 0, run.episodes, "history empty")
    episode_range = last_record.get("episode_range") or {}
    final_episode = int(episode_range.get("final") or 0)
    status = str(last_record.get("status") or "unknown")
    if status == "completed" and final_episode >= run.episodes:
        return RunStatus("completed", final_episode, run.episodes, "history completed")
    return RunStatus("incomplete", final_episode, run.episodes, f"history status={status}")


def _last_metric_episode(output: Path) -> int:
    if not output.exists():
        return 0
    last_episode = 0
    with output.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                last_episode = max(last_episode, int(row.get("episode") or 0))
            except ValueError:
                continue
    return last_episode


def _print_statuses(statuses: list[tuple[RunConfig, RunStatus]]) -> None:
    for run, status in statuses:
        _print_status(run, status)


def _print_status(run: RunConfig, status: RunStatus) -> None:
    print(
        f"{run.run_name:<22} {status.state:<12} ep={status.episode}/{status.requested} "
        f"config={run.path} ({status.reason})",
        flush=True,
    )


def _remove_run_artifacts(run: RunConfig, *, dry_run: bool) -> None:
    targets = [
        run.output,
        run.eval_output,
        run.history_output,
        run.model_output,
        run.checkpoint_path.parent,
    ]
    for target in targets:
        if not target.exists():
            continue
        _assert_workspace_artifact(target)
        prefix = "would delete" if dry_run else "delete"
        print(f"  -> {prefix}: {target}", flush=True)
        if dry_run:
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def _assert_workspace_artifact(path: Path) -> None:
    workspace = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise SystemExit(f"Refusing to delete outside workspace: {resolved}") from exc
    if not (resolved.parts and any(part in {"experiments", "models"} for part in resolved.parts)):
        raise SystemExit(f"Refusing to delete non-artifact path: {resolved}")


def _module_for_config(config: Path) -> str:
    for part in config.parts:
        if part in MODULES:
            return MODULES[part]
    raise SystemExit(f"Cannot infer training module from config path: {config}")


def _algorithm_from_path(config: Path) -> str:
    for part in config.parts:
        if part in MODULES:
            return part
    raise SystemExit(f"Cannot infer algorithm from config path: {config}")


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        try:
            return abs(float(left) - right) < 1e-9
        except (TypeError, ValueError):
            return False
    return left == right


def _checkpoint_episode(path: Path) -> int:
    match = re.search(r"_ep(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
