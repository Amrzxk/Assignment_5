"""Assignment report generator for SeqTrack evaluations.

Reads the aggregated evaluation results produced by ``evaluate_checkpoints.py``
and generates the required visualizations and markdown report for submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pickle

from lib.test.evaluation.environment import env_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SeqTrack evaluation report and plots.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to evaluation results JSON. Relative paths are resolved against the testing directory.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix to use for generated artifacts (defaults to the input filename stem).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store outputs. Defaults to the testing directory.",
    )
    return parser.parse_args()


def load_results(results_path: Path) -> Dict:
    if not results_path.is_file():
        raise FileNotFoundError(f"Evaluation results not found at {results_path}")

    with results_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def prepare_metrics(results: Dict) -> Dict[str, List[float]]:
    epochs_sorted = sorted(results["epochs"], key=lambda entry: entry["epoch"])

    epochs = [entry["epoch"] for entry in epochs_sorted]
    iou = [entry["metrics"].get("IoU", 0.0) for entry in epochs_sorted]
    precision = [entry["metrics"].get("Precision", 0.0) for entry in epochs_sorted]
    auc = [entry["metrics"].get("AUC", 0.0) for entry in epochs_sorted]
    fps = [entry["inference"].get("fps", 0.0) for entry in epochs_sorted]
    ms_per_frame = [entry["inference"].get("ms_per_frame", 0.0) for entry in epochs_sorted]

    return {
        "epochs": epochs,
        "IoU": iou,
        "Precision": precision,
        "AUC": auc,
        "FPS": fps,
        "ms_per_frame": ms_per_frame,
    }


def plot_metrics(metrics: Dict[str, List[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics["epochs"], metrics["IoU"], marker="o", label="IoU (%)")
    plt.plot(metrics["epochs"], metrics["Precision"], marker="s", label="Precision (%)")
    plt.plot(metrics["epochs"], metrics["AUC"], marker="^", label="AUC (%)")

    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title("SeqTrack Evaluation Metrics vs Epoch")
    plt.xticks(metrics["epochs"])
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_single_metric(metrics: Dict[str, List[float]], key: str, ylabel: str,
                       title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics["epochs"], metrics[key], marker="o")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(metrics["epochs"])
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line] + row_lines)


def write_report(
    report_path: Path,
    metrics: Dict[str, List[float]],
    summary_graph: Path,
    auc_graph: Path,
    fps_graph: Path,
    class_iou_graph: Optional[Path],
    class_auc_graph: Optional[Path],
    class_precision_graph: Optional[Path],
    class_metrics: Optional[Dict[str, Any]]
) -> None:
    report_lines: List[str] = []

    report_lines.append("# Assignment 4: Evaluation Report")
    report_lines.append("\n## Inference Tables\n")

    inference_rows = [
        [
            f"{epoch}",
            f"{fps:.2f}",
            f"{ms:.2f}",
        ]
        for epoch, fps, ms in zip(metrics["epochs"], metrics["FPS"], metrics["ms_per_frame"])
    ]
    inference_table = format_table(["Epoch", "FPS", "ms/frame"], inference_rows)
    report_lines.append("### Table 1: Inference Rate\n")
    report_lines.append(inference_table + "\n")

    evaluation_rows = [
        [
            f"{epoch}",
            f"{iou:.2f}",
            f"{precision:.2f}",
            f"{auc:.2f}",
        ]
        for epoch, iou, precision, auc in zip(
            metrics["epochs"], metrics["IoU"], metrics["Precision"], metrics["AUC"]
        )
    ]
    evaluation_table = format_table(["Epoch", "IoU (%)", "Precision (%)", "AUC (%)"], evaluation_rows)
    report_lines.append("### Table 2: Evaluation Results\n")
    report_lines.append(evaluation_table + "\n")

    report_lines.append("## Evaluation Graphs\n")
    report_lines.append(f"![Overall Metrics]({summary_graph.name})\n")
    report_lines.append(f"![AUC vs Epoch]({auc_graph.name})\n")
    report_lines.append(f"![Inference FPS vs Epoch]({fps_graph.name})\n")

    if class_metrics is not None:
        classes = class_metrics["classes"]

        report_lines.append("## Class-wise Metrics\n")
        headers = ["Epoch"]
        for cls in classes:
            headers.extend([f"{cls} IoU (%)", f"{cls} Precision (%)", f"{cls} AUC (%)"])

        class_rows: List[List[str]] = []
        epochs = metrics["epochs"]
        for idx, epoch in enumerate(epochs):
            row = [f"{epoch}"]
            for cls in classes:
                row.append(f"{class_metrics['IoU'][cls][idx]:.2f}")
                row.append(f"{class_metrics['Precision'][cls][idx]:.2f}")
                row.append(f"{class_metrics['AUC'][cls][idx]:.2f}")
            class_rows.append(row)

        report_lines.append("### Table 3: Class-wise Results\n")
        report_lines.append(format_table(headers, class_rows) + "\n")

        if class_iou_graph is not None:
            report_lines.append(f"![Class IoU vs Epoch]({class_iou_graph.name})\n")
        if class_auc_graph is not None:
            report_lines.append(f"![Class AUC vs Epoch]({class_auc_graph.name})\n")
        if class_precision_graph is not None:
            report_lines.append(f"![Class Precision vs Epoch]({class_precision_graph.name})\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def get_eval_data(results: Dict, testing_dir: Path) -> Optional[Dict]:
    report_name = results.get("report_name")
    if not report_name:
        return None

    eval_data_path = testing_dir / "result_plots" / report_name / "eval_data.pkl"
    if not eval_data_path.is_file():
        return None

    with eval_data_path.open("rb") as fh:
        return pickle.load(fh)


def compute_class_metrics(eval_data: Dict) -> Optional[Dict[str, Any]]:
    if eval_data is None:
        return None

    sequences = eval_data["sequences"]
    classes = [seq.split("-")[0] for seq in sequences]
    unique_classes = sorted(set(classes))

    valid_sequence = np.array(eval_data["valid_sequence"], dtype=bool)
    if not valid_sequence.any():
        return None

    avg_overlap_all = np.array(eval_data["avg_overlap_all"])  # shape (seq, tracker)
    success_overlap = np.array(eval_data["ave_success_rate_plot_overlap"])  # shape (seq, tracker, thresholds)
    success_center = np.array(eval_data["ave_success_rate_plot_center"])  # shape (seq, tracker, thresholds)
    thresholds_center = np.array(eval_data["threshold_set_center"])  # length T

    # Index for 20px precision
    try:
        precision_idx = int(np.where(thresholds_center == 20)[0][0])
    except IndexError:
        precision_idx = None

    class_metrics: Dict[str, Any] = {
        "classes": unique_classes,
        "IoU": {},
        "Precision": {},
        "AUC": {},
    }

    num_trackers = avg_overlap_all.shape[1]

    for cls in unique_classes:
        class_mask = (np.array(classes) == cls) & valid_sequence
        if not class_mask.any():
            continue

        iou_values = []
        precision_values = []
        auc_values = []

        for tracker_idx in range(num_trackers):
            # IoU (AO) per class
            class_iou = avg_overlap_all[class_mask, tracker_idx].mean() * 100.0
            iou_values.append(float(class_iou))

            # Success overlap (AUC)
            class_success_overlap = success_overlap[class_mask, tracker_idx, :].mean(axis=0)
            class_auc = class_success_overlap.mean() * 100.0
            auc_values.append(float(class_auc))

            # Precision at 20px
            if precision_idx is not None:
                class_precision = success_center[class_mask, tracker_idx, precision_idx].mean() * 100.0
            else:
                class_precision = float("nan")
            precision_values.append(float(class_precision))

        class_metrics["IoU"][cls] = iou_values
        class_metrics["Precision"][cls] = precision_values
        class_metrics["AUC"][cls] = auc_values

    return class_metrics


def plot_class_metrics(
    epochs: List[int],
    class_metrics: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path
) -> None:
    plt.figure(figsize=(8, 5))
    for cls in class_metrics["classes"]:
        if cls not in class_metrics[metric_key]:
            continue
        values = class_metrics[metric_key][cls]
        plt.plot(epochs, values, marker="o", label=cls)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fps(metrics: Dict[str, List[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics["epochs"], metrics["FPS"], marker="o", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Frames per Second")
    plt.title("Inference FPS vs Epoch")
    plt.xticks(metrics["epochs"])
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()

    settings = env_settings()
    save_dir = Path(settings.save_dir)
    testing_dir = save_dir / "testing"

    if args.input is not None:
        results_path = Path(args.input)
        if not results_path.is_absolute():
            results_path = testing_dir / results_path
    else:
        results_path = testing_dir / "evaluation_results.json"

    results = load_results(results_path)
    metrics = prepare_metrics(results)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = testing_dir / output_dir
    else:
        output_dir = testing_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    default_prefix = results_path.stem.replace(".json", "")
    output_prefix = args.output_prefix or default_prefix

    summary_graph_path = output_dir / f"{output_prefix}_metrics.png"
    auc_graph_path = output_dir / f"{output_prefix}_auc.png"
    fps_graph_path = output_dir / f"{output_prefix}_fps.png"
    class_iou_graph_path = output_dir / f"{output_prefix}_class_iou.png"
    class_auc_graph_path = output_dir / f"{output_prefix}_class_auc.png"
    class_precision_graph_path = output_dir / f"{output_prefix}_class_precision.png"
    report_path = output_dir / f"{output_prefix}_report.md"

    # Overall graphs
    plot_metrics(metrics, summary_graph_path)
    plot_single_metric(metrics, "AUC", "AUC (%)", "AUC vs Epoch", auc_graph_path)
    plot_fps(metrics, fps_graph_path)

    eval_data = get_eval_data(results, testing_dir)
    class_metrics = compute_class_metrics(eval_data)

    if class_metrics is not None and class_metrics["classes"]:
        plot_class_metrics(
            metrics["epochs"], class_metrics, "IoU",
            "Class-wise IoU vs Epoch", "IoU (%)", class_iou_graph_path
        )
        plot_class_metrics(
            metrics["epochs"], class_metrics, "AUC",
            "Class-wise AUC vs Epoch", "AUC (%)", class_auc_graph_path
        )
        plot_class_metrics(
            metrics["epochs"], class_metrics, "Precision",
            "Class-wise Precision vs Epoch", "Precision (%)", class_precision_graph_path
        )
    else:
        class_iou_graph_path = None
        class_auc_graph_path = None
        class_precision_graph_path = None

    write_report(
        report_path,
        metrics,
        summary_graph_path,
        auc_graph_path,
        fps_graph_path,
        class_iou_graph_path,
        class_auc_graph_path,
        class_precision_graph_path,
        class_metrics,
    )

    print(f"Summary graph saved to {summary_graph_path}")
    print(f"AUC graph saved to {auc_graph_path}")
    print(f"FPS graph saved to {fps_graph_path}")
    if class_iou_graph_path is not None:
        print(f"Class IoU graph saved to {class_iou_graph_path}")
    if class_auc_graph_path is not None:
        print(f"Class AUC graph saved to {class_auc_graph_path}")
    if class_precision_graph_path is not None:
        print(f"Class Precision graph saved to {class_precision_graph_path}")
    print(f"Markdown report saved to {report_path}")


if __name__ == "__main__":
    main()


