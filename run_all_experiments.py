from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    Path("configs/q_learning/q_learning_lr_001.yaml"),
    Path("configs/dqn/dqn_lr_001.yaml"),
    Path("configs/double_dqn/double_dqn_lr_001.yaml"),
    Path("configs/q_learning/q_learning_lr_01.yaml"),
    Path("configs/dqn/dqn_lr_01.yaml"),
    Path("configs/double_dqn/double_dqn_lr_01.yaml"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pacman training experiments from YAML configs.")
    parser.add_argument("--all-lr", action="store_true", help="Run all 9 learning-rate configs.")
    parser.add_argument("--resume", action="store_true", help="Resume every selected run from its latest checkpoint.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip compare_runs after training.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--plot-output", default="experiments/plots/training_comparison.png")
    parser.add_argument("configs", nargs="*", help="Optional explicit config paths to run.")
    args = parser.parse_args()

    configs = [Path(config) for config in args.configs]
    if not configs:
        configs = LR_SWEEP_CONFIGS if args.all_lr else MAIN_CONFIGS

    for config in configs:
        module = _module_for_config(config)
        command = [sys.executable, "-m", module, "--config", str(config)]
        if args.resume:
            command.append("--resume")
        _run(command, dry_run=args.dry_run)

    if not args.skip_plots:
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


def _module_for_config(config: Path) -> str:
    for part in config.parts:
        if part in MODULES:
            return MODULES[part]
    raise SystemExit(f"Cannot infer training module from config path: {config}")


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
