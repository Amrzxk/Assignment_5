from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from lib.test.evaluation.environment import env_settings


@dataclass
class EvaluationRun:
    label: str
    file_path: Path
    epochs: List[int]
    metrics: Dict[str, List[float]]

    def metric_map(self, key: str) -> Dict[int, float]:
        return {epoch: self.metrics[key][idx] for idx, epoch in enumerate(self.epochs)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple SeqTrack evaluation runs across checkpoints."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of evaluation_results JSON files to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each input. Must match the number of inputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store comparison artifacts (defaults to testing directory).",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="comparison",
        help="Filename prefix for generated plots and tables.",
    )
    return parser.parse_args()


def load_evaluation(path: Path, label: Optional[str] = None) -> EvaluationRun:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    epochs_sorted = sorted(data["epochs"], key=lambda entry: entry["epoch"])
    epochs = [entry["epoch"] for entry in epochs_sorted]
    metrics = {
        "IoU": [entry["metrics"].get("IoU", float("nan")) for entry in epochs_sorted],
        "Precision": [entry["metrics"].get("Precision", float("nan")) for entry in epochs_sorted],
        "AUC": [entry["metrics"].get("AUC", float("nan")) for entry in epochs_sorted],
        "FPS": [entry["inference"].get("fps", float("nan")) for entry in epochs_sorted],
        "ms_per_frame": [entry["inference"].get("ms_per_frame", float("nan")) for entry in epochs_sorted],
    }

    derived_label = label
    if derived_label is None:
        derived_label = (
            data.get("metadata", {}).get("hf_folder")
            or data.get("output_name")
            or path.stem
        )

    return EvaluationRun(
        label=derived_label,
        file_path=path,
        epochs=epochs,
        metrics=metrics,
    )


def plot_metric(all_epochs: List[int], runs: List[EvaluationRun], key: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "P", "X", "*", "v"]

    for idx, run in enumerate(runs):
        values_map = run.metric_map(key)
        values = [values_map.get(epoch, np.nan) for epoch in all_epochs]
        marker = markers[idx % len(markers)]
        plt.plot(all_epochs, values, marker=marker, label=run.label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison")
    plt.xticks(all_epochs)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line] + row_lines)


def build_combined_table(all_epochs: List[int], runs: List[EvaluationRun]) -> str:
    headers = ["Epoch"]
    metric_keys = ["IoU", "Precision", "AUC", "FPS"]
    for run in runs:
        for key in metric_keys:
            headers.append(f"{run.label} {key}")

    rows: List[List[str]] = []
    for epoch in all_epochs:
        row = [f"{epoch}"]
        for run in runs:
            values_map = run.metric_map("IoU")
            precision_map = run.metric_map("Precision")
            auc_map = run.metric_map("AUC")
            fps_map = run.metric_map("FPS")
            row.extend(
                [
                    f"{values_map.get(epoch, float('nan')):.2f}",
                    f"{precision_map.get(epoch, float('nan')):.2f}",
                    f"{auc_map.get(epoch, float('nan')):.2f}",
                    f"{fps_map.get(epoch, float('nan')):.2f}",
                ]
            )
        rows.append(row)

    return format_markdown_table(headers, rows)


def main() -> None:
    args = parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must have the same length as --inputs.")

    settings = env_settings()
    save_dir = Path(settings.save_dir)
    testing_dir = save_dir / "testing"

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = testing_dir / output_dir
    else:
        output_dir = testing_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: List[EvaluationRun] = []
    for idx, input_path in enumerate(args.inputs):
        path = Path(input_path)
        if not path.is_absolute():
            path = testing_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation file not found: {path}")

        label = args.labels[idx] if args.labels else None
        runs.append(load_evaluation(path, label))

    all_epochs = sorted({epoch for run in runs for epoch in run.epochs})

    metric_definitions = [
        ("IoU", "IoU (%)"),
        ("Precision", "Precision (%)"),
        ("AUC", "AUC (%)"),
        ("FPS", "Frames Per Second"),
    ]

    for key, ylabel in metric_definitions:
        plot_metric(
            all_epochs,
            runs,
            key,
            ylabel,
            output_dir / f"{args.output_prefix}_{key.lower()}_comparison.png",
        )

    table_markdown = build_combined_table(all_epochs, runs)
    table_path = output_dir / f"{args.output_prefix}_comparison_table.md"
    table_path.write_text(table_markdown, encoding="utf-8")

    summary_path = output_dir / f"{args.output_prefix}_summary.txt"
    summary_lines = [
        "SeqTrack Checkpoint Comparison",
        f"Inputs: {', '.join(str(run.file_path) for run in runs)}",
        f"Labels: {', '.join(run.label for run in runs)}",
        f"Epochs compared: {all_epochs}",
        f"Markdown table: {table_path}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Comparison complete.")
    for key, _ in metric_definitions:
        print(f"  Saved {key} plot to {output_dir / f'{args.output_prefix}_{key.lower()}_comparison.png'}")
    print(f"  Markdown table written to {table_path}")
    print(f"  Summary written to {summary_path}")


if __name__ == "__main__":
    main()


