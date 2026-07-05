from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_COLUMNS = {
    "reward": ("reward", "avg_reward"),
    "completion": ("completion_rate", "avg_completion_rate"),
    "win_rate": ("win_rate",),
    "steps": ("steps", "avg_steps"),
    "loss": ("loss",),
}

METRIC_LABELS = {
    "reward": "Reward",
    "completion": "Completion rate",
    "win_rate": "Win rate",
    "steps": "Steps",
    "loss": "Loss",
}

METRIC_ALGORITHMS = {
    "loss": {"dqn", "double_dqn"},
}


@dataclass(frozen=True)
class MetricFile:
    path: Path
    algorithm: str
    lr_name: str
    is_eval: bool

    @property
    def label(self) -> str:
        label = self.algorithm
        if self.is_eval:
            label += " eval"
        return label


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


def discover_metric_files(source: str) -> list[MetricFile]:
    metrics_dir = Path("experiments/metrics")
    if not metrics_dir.exists():
        print(f"No files provided and {metrics_dir} does not exist.")
        return []

    metric_files = [
        metric_file
        for path in sorted(metrics_dir.rglob("*_metrics.csv"))
        if (metric_file := parse_metric_file(path)) is not None
    ]
    if source == "train":
        metric_files = [metric_file for metric_file in metric_files if not metric_file.is_eval]
    elif source == "eval":
        metric_files = [metric_file for metric_file in metric_files if metric_file.is_eval]

    print(f"Auto-discovered {len(metric_files)} metric files:")
    for metric_file in metric_files:
        print(f"  - {metric_file.path}")
    return metric_files


def parse_metric_file(path: Path) -> MetricFile | None:
    algorithm = path.parent.name
    stem = path.stem
    if stem.endswith("_eval_metrics"):
        run_name = stem.removesuffix("_eval_metrics")
        is_eval = True
    elif stem.endswith("_metrics"):
        run_name = stem.removesuffix("_metrics")
        is_eval = False
    else:
        return None

    prefix = f"{algorithm}_"
    if not run_name.startswith(prefix):
        return None
    lr_name = run_name.removeprefix(prefix)
    return MetricFile(path=path, algorithm=algorithm, lr_name=lr_name, is_eval=is_eval)


def metric_files_from_paths(paths: list[str]) -> list[MetricFile]:
    metric_files: list[MetricFile] = []
    for file_name in paths:
        path = Path(file_name)
        metric_file = parse_metric_file(path)
        if metric_file is None:
            algorithm = path.parent.name or "unknown"
            metric_file = MetricFile(path=path, algorithm=algorithm, lr_name="custom", is_eval=False)
        metric_files.append(metric_file)
    return metric_files


def group_by_lr(metric_files: list[MetricFile]) -> dict[str, list[MetricFile]]:
    groups: dict[str, list[MetricFile]] = {}
    for metric_file in metric_files:
        groups.setdefault(metric_file.lr_name, []).append(metric_file)
    return dict(sorted(groups.items()))


def output_for_lr_metric(output: Path, lr_name: str, metric: str, source: str) -> Path:
    lr_label = lr_name.removeprefix("lr_")
    source_part = "" if source == "train" else f"{source}_"
    return output.with_name(f"{source_part}{metric}_{lr_label}{output.suffix}")


def output_for_combined(output: Path, source: str) -> Path:
    if source == "train":
        return output
    return output.with_name(f"{output.stem}_{source}{output.suffix}")


def plot_metric_files(
    metric_files: list[MetricFile],
    metrics: list[str],
    window: int,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        plotted = False
        eligible_algorithms = METRIC_ALGORITHMS.get(metric)
        eligible_metric_files = [
            metric_file
            for metric_file in metric_files
            if eligible_algorithms is None or metric_file.algorithm in eligible_algorithms
        ]
        for metric_file in sorted(eligible_metric_files, key=lambda item: (item.algorithm, item.is_eval)):
            if not metric_file.path.exists():
                continue
            episodes, values = read_metric(metric_file.path, metric)
            if not values:
                continue
            linestyle = "--" if metric_file.is_eval else "-"
            axis.plot(
                episodes,
                moving_average(values, window),
                label=metric_file.label,
                linestyle=linestyle,
            )
            plotted = True
        axis.set_ylabel(f"{METRIC_LABELS[metric]} MA({window})")
        if plotted:
            axis.legend()
        else:
            axis.text(0.5, 0.5, f"No {metric} data", transform=axis.transAxes, ha="center")

    axes[0].set_title(title)
    axes[-1].set_xlabel("Episode")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reward, completion and win-rate comparisons for Pacman training runs."
    )
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--output", default="experiments/plots/training_comparison.png")
    parser.add_argument(
        "--source",
        choices=["train", "eval", "all"],
        default="train",
        help="Metric files to plot when auto-discovering files.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Write one combined plot instead of one plot per learning rate.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_COLUMNS),
        default=["reward", "completion", "win_rate", "steps", "loss"],
    )
    parser.add_argument("files", nargs="*", default=[])
    args = parser.parse_args()

    output_path = Path(args.output)
    metric_files = metric_files_from_paths(args.files) if args.files else discover_metric_files(args.source)
    if not metric_files:
        raise SystemExit("No metric files to plot.")

    if args.combined or args.files:
        plot_metric_files(
            metric_files=metric_files,
            metrics=args.metrics,
            window=args.window,
            output_path=output_for_combined(output_path, args.source),
            title="Training comparison",
        )
        return

    for lr_name, lr_metric_files in group_by_lr(metric_files).items():
        for metric in args.metrics:
            plot_metric_files(
                metric_files=lr_metric_files,
                metrics=[metric],
                window=args.window,
                output_path=output_for_lr_metric(output_path, lr_name, metric, args.source),
                title=f"Learning rate {lr_name.replace('lr_', '0.')}",
            )


if __name__ == "__main__":
    main()
