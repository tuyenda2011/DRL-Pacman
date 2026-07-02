import argparse
import json
from pathlib import Path

from src.training.logging_utils import MetricsCsvLogger, print_saved_outputs, read_metric_rows
from src.training.logging_utils import append_training_run_log, should_log_episode


def test_metrics_csv_logger_streams_rows(tmp_path) -> None:
    output = tmp_path / "metrics.csv"
    fieldnames = ["episode", "reward", "event"]

    with MetricsCsvLogger(output, fieldnames) as logger:
        logger.write({"episode": 1, "reward": 1.5, "event": "timeout"})

    assert read_metric_rows(output) == [
        {"episode": 1, "reward": 1.5, "event": "timeout"},
    ]


def test_metrics_csv_logger_rewrites_existing_resume_rows(tmp_path) -> None:
    output = tmp_path / "metrics.csv"
    fieldnames = ["episode", "reward", "event"]

    with MetricsCsvLogger(output, fieldnames) as logger:
        logger.write({"episode": 1, "reward": 1.0, "event": "timeout"})
        logger.write({"episode": 2, "reward": 2.0, "event": "win"})

    resume_rows = read_metric_rows(output, up_to_episode=1)
    with MetricsCsvLogger(output, fieldnames, existing_rows=resume_rows) as logger:
        logger.write({"episode": 2, "reward": 3.0, "event": "timeout"})

    assert read_metric_rows(output) == [
        {"episode": 1, "reward": 1.0, "event": "timeout"},
        {"episode": 2, "reward": 3.0, "event": "timeout"},
    ]


def test_append_training_run_log_records_completed_run(tmp_path) -> None:
    output = tmp_path / "training_runs.jsonl"
    args = argparse.Namespace(
        episodes=10,
        output="experiments/metrics/q_learning_metrics.csv",
        model_output="models/final/q_learning/q_learning.pkl",
        checkpoint_path="models/checkpoints/q_learning/q_learning_checkpoint.pkl",
        history_output=str(output),
        resume=False,
        seed=42,
    )

    append_training_run_log(
        output,
        algorithm="Q-learning",
        args=args,
        status="completed",
        start_episode=1,
        final_episode=10,
        final_metrics={"episode": 10, "reward": 12.5, "event": "win"},
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["algorithm"] == "Q-learning"
    assert records[0]["episode_range"]["trained"] == 10
    assert Path(records[0]["artifacts"]["model"]).as_posix() == "models/final/q_learning/q_learning.pkl"
    assert records[0]["final_metrics"]["event"] == "win"


def test_interrupted_episode_is_always_logged() -> None:
    assert should_log_episode(
        episode=7,
        total_episodes=100,
        log_interval=0,
        event="interrupted",
    )


def test_interrupted_training_does_not_report_saved_model(capsys) -> None:
    print_saved_outputs(
        "experiments/metrics/q_learning_metrics.csv",
        None,
        "experiments/history/training_runs.jsonl",
        status="interrupted",
    )

    captured = capsys.readouterr()
    assert "Training interrupted." in captured.out
    assert "Model saved to:   not saved" in captured.out
