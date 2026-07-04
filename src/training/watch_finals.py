from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ALGORITHMS = ("q_learning", "dqn", "double_dqn")
MODEL_EXTENSIONS = {
    "q_learning": "pkl",
    "dqn": "pt",
    "double_dqn": "pt",
}


@dataclass(frozen=True)
class FinalModel:
    index: int
    algorithm: str
    run_name: str
    model_path: Path
    config_path: Path | None

    @property
    def label(self) -> str:
        lr_label = self.run_name.replace(f"{self.algorithm}_", "")
        return f"{self.algorithm:<10} {lr_label:<8} {self.model_path}"


def discover_final_models(
    algorithm: str | None,
    models_root: Path,
    configs_root: Path,
) -> list[FinalModel]:
    algorithms = [algorithm] if algorithm else list(ALGORITHMS)
    models: list[FinalModel] = []

    for algorithm_name in algorithms:
        extension = MODEL_EXTENSIONS[algorithm_name]
        algorithm_dir = models_root / algorithm_name
        for model_path in sorted(algorithm_dir.glob(f"*.{extension}")):
            run_name = model_path.stem
            config_path = configs_root / algorithm_name / f"{run_name}.yaml"
            models.append(
                FinalModel(
                    index=0,
                    algorithm=algorithm_name,
                    run_name=run_name,
                    model_path=model_path,
                    config_path=config_path if config_path.exists() else None,
                )
            )

    return [
        FinalModel(
            index=index,
            algorithm=model.algorithm,
            run_name=model.run_name,
            model_path=model.model_path,
            config_path=model.config_path,
        )
        for index, model in enumerate(models, start=1)
    ]


def print_models(models: Sequence[FinalModel]) -> None:
    if not models:
        print("No final models found.")
        return

    print("Final models:")
    for model in models:
        config_status = "" if model.config_path else "  (missing config; using viewer defaults)"
        print(f"{model.index}. {model.label}{config_status}")


def choose_models(models: Sequence[FinalModel]) -> list[FinalModel]:
    print_models(models)
    print("a. Run all")
    choice = input("Choose model: ").strip().lower()

    if choice in {"a", "all"}:
        return list(models)

    try:
        index = int(choice)
    except ValueError as exc:
        raise SystemExit(f"Invalid choice: {choice}") from exc

    for model in models:
        if model.index == index:
            return [model]

    raise SystemExit(f"Model index out of range: {index}")


def build_watch_command(model: FinalModel, args: argparse.Namespace, viewer_args: Sequence[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.training.watch_model",
        "--algorithm",
        model.algorithm,
        "--run-name",
        model.run_name,
        "--model-path",
        str(model.model_path),
        "--episodes",
        str(args.episodes),
        "--delay-ms",
        str(args.delay_ms),
        "--cell-size",
        str(args.cell_size),
    ]
    if model.config_path:
        command.extend(["--config", str(model.config_path)])
    command.extend(viewer_args)
    return command


def run_model(model: FinalModel, args: argparse.Namespace, viewer_args: Sequence[str]) -> None:
    command = build_watch_command(model, args, viewer_args)
    print(f"\nLaunching {model.algorithm} / {model.run_name}")
    print(" ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pick and watch final Mini Pacman models from models/final/.",
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, default=None)
    parser.add_argument("--all", action="store_true", help="Run all discovered final models in order.")
    parser.add_argument("--list", action="store_true", help="List discovered final models without launching GUI.")
    parser.add_argument("--dry-run", action="store_true", help="Print watch_model commands without running them.")
    parser.add_argument("--models-root", default="models/final")
    parser.add_argument("--configs-root", default="configs")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--delay-ms", type=int, default=500)
    parser.add_argument("--cell-size", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, viewer_args = parser.parse_known_args(argv)

    models = discover_final_models(
        algorithm=args.algorithm,
        models_root=Path(args.models_root),
        configs_root=Path(args.configs_root),
    )
    if not models:
        raise SystemExit(f"No final models found under {Path(args.models_root)}")

    if args.list:
        print_models(models)
        return

    selected_models = list(models) if args.all else choose_models(models)
    for model in selected_models:
        run_model(model, args, viewer_args)


if __name__ == "__main__":
    main()
