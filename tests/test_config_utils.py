import argparse

from src.training.config_utils import parse_args_with_config


def test_parse_args_with_config_uses_yaml_defaults(tmp_path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        "\n".join(
            [
                "algorithm: ignored",
                "episodes: 123",
                "learning_rate: 0.25",
                "ghost_count: 2",
            ]
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--ghost-count", type=int, default=3)

    args = parse_args_with_config(parser, str(config), argv=[])

    assert args.episodes == 123
    assert args.learning_rate == 0.25
    assert args.ghost_count == 2
    assert args.config == str(config)
    assert args.algorithm == "ignored"
    assert args.run_name == "train"


def test_cli_flags_override_config_defaults(tmp_path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("episodes: 123\nghost_count: 2\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ghost-count", type=int, default=3)

    args = parse_args_with_config(
        parser,
        str(config),
        argv=["--episodes", "7"],
    )

    assert args.episodes == 7
    assert args.ghost_count == 2
    assert args.algorithm == "unknown"
    assert args.run_name == "train"


def test_string_outputs_can_use_run_name_and_algorithm_placeholders(tmp_path) -> None:
    config = tmp_path / "dqn_lr_001.yaml"
    config.write_text("algorithm: dqn\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/{algorithm}/{run_name}.csv")

    args = parse_args_with_config(parser, str(config), argv=[])

    assert args.output == "experiments/dqn/dqn_lr_001.csv"
