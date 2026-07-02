from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_COLUMNS = {
    "reward": ("reward", "avg_reward"),
    "completion": ("completion_rate", "avg_completion_rate"),
    "win_rate": ("win_rate",),
}

METRIC_LABELS = {
    "reward": "Reward",
    "completion": "Completion rate",
    "win_rate": "Win rate",
}


def read_metric(path: Path, metric: str) -> tuple[list[int], list[float]]:
    episodes: list[int] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        metric_column = _metric_column(reader.fieldnames or [], metric)
        if metric_column is None:
            return episodes, values
        for row in reader:
            episodes.append(int(row["episode"]))
            values.append(float(row[metric_column]))
    return episodes, values


def moving_average(values: list[float], window: int) -> list[float]:
    averaged: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(sum(values[start : index + 1]) / (index - start + 1))
    return averaged


def _metric_column(fieldnames: list[str], metric: str) -> str | None:
    for column in METRIC_COLUMNS[metric]:
        if column in fieldnames:
            return column
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reward, completion and win-rate comparisons for Pacman training runs."
    )
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--output", default="experiments/plots/training_comparison.png")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_COLUMNS),
        default=["reward", "completion", "win_rate"],
    )
    parser.add_argument("files", nargs="*", default=[])
    args = parser.parse_args()

    if not args.files:
        metrics_dir = Path("experiments/metrics")
        if metrics_dir.exists():
            args.files = sorted([str(p) for p in metrics_dir.rglob("*_metrics.csv")])
            print(f"Auto-discovered {len(args.files)} metric files:")
            for f in args.files:
                print(f"  - {f}")
        else:
            print(f"No files provided and {metrics_dir} does not exist.")

    fig, axes = plt.subplots(len(args.metrics), 1, figsize=(10, 3.5 * len(args.metrics)), sharex=True)
    if len(args.metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, args.metrics):
        plotted = False
        for file_name in args.files:
            path = Path(file_name)
            if not path.exists():
                continue
            episodes, values = read_metric(path, metric)
            if not values:
                continue
            label = path.stem.replace("_metrics", "").replace("_eval", " eval")
            axis.plot(episodes, moving_average(values, args.window), label=label)
            plotted = True
        axis.set_ylabel(f"{METRIC_LABELS[metric]} MA({args.window})")
        if plotted:
            axis.legend()
        else:
            axis.text(0.5, 0.5, f"No {metric} data", transform=axis.transAxes, ha="center")

    axes[-1].set_xlabel("Episode")
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


if __name__ == "__main__":
    main()
