from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    default_config: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser.add_argument(
        "--config",
        default=default_config,
        help="YAML config file used as parser defaults. Command-line flags override it.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name of the run. Defaults to the base name of the config file.",
    )
    parser.add_argument(
        "--algorithm",
        default="unknown",
        help="Algorithm name, usually loaded from YAML config.",
    )
    config_args = _parse_config_arg(default_config, argv)
    parser.set_defaults(config=config_args.config)

    config_defaults = load_config_defaults(config_args.config, parser)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)

    if args.run_name is None:
        args.run_name = Path(args.config).stem

    # Auto-format string arguments that contain {run_name} or {algorithm}
    algorithm = getattr(args, "algorithm", "unknown")
    for key, value in vars(args).items():
        if isinstance(value, str):
            try:
                formatted = value.format(run_name=args.run_name, algorithm=algorithm)
                setattr(args, key, formatted)
            except (KeyError, ValueError):
                pass  # Ignore strings with unresolvable placeholders

    return args


def load_config_defaults(config_path: str | Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {path}")

    valid_dests = {action.dest for action in parser._actions}
    defaults: dict[str, Any] = {}
    for key, value in config.items():
        dest = str(key).replace("-", "_")
        if dest in valid_dests:
            defaults[dest] = value
    return defaults


def _parse_config_arg(
    default_config: str,
    argv: Sequence[str] | None,
) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=default_config)
    config_args, _ = config_parser.parse_known_args(argv)
    return config_args
