"""
Generate evaluation tables and plots directly from raw SeqTrack result folders.

The script walks the per-epoch result directories (e.g. ``seqtrack_b256_005``),
loads the predicted bounding boxes and timing information for every sequence,
compares them against the LaSOT ground-truth annotations, and aggregates the
standard tracking metrics (Average Overlap/IoU, Precision @20px, and Success AUC).

Outputs:
    * ``testing/testing_report.md`` – Markdown report with metric and inference tables
    * ``testing/testing_metrics.png`` – Combined IoU / Precision / AUC plot
    * ``testing/testing_auc.png`` – Individual AUC vs epoch plot
    * ``testing/testing_fps.png`` – Inference FPS vs epoch plot

Example:
    python generate_testing_report.py \\
        --dataset-root "E:/datasets/LaSOT/LaSOTBenchmark/train" \\
        --results-root "testing/results/seqtrack" \\
        --epochs 5 10 15 20 80 \\
        --exclude-epochs 85
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import os


# -----------------------------------------------------------------------------#
# Data containers


@dataclass
class SequenceStats:
    sequence: str
    class_name: str
    num_frames: int
    valid_frames: int
    iou_curve: np.ndarray  # success curve over IoU thresholds
    precision_curve: np.ndarray  # precision curve over center thresholds
    iou_mean: float  # Average overlap (0-1)
    precision_at_20: float  # fraction within 20px
    auc: float  # mean success curve value
    total_time_sec: float


@dataclass
class EpochStats:
    epoch: int
    num_sequences: int
    total_frames: int
    total_time_sec: float
    iou_mean: float  # in percentage
    precision_mean: float  # in percentage
    auc_mean: float  # in percentage
    fps: float
    ms_per_frame: float


# -----------------------------------------------------------------------------#
# Utility helpers


IOU_THRESHOLDS = np.linspace(0.0, 1.0, 21, dtype=np.float64)
PRECISION_THRESHOLDS = np.arange(0, 51, dtype=np.float64)
PRECISION_TARGET_PX = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate SeqTrack evaluation metrics from raw result folders."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root directory of the LaSOT dataset (class folders containing groundtruth.txt files).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("testing/results/seqtrack"),
        help="Root directory containing per-epoch result folders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of epochs to process. When omitted, inferred from folder names.",
    )
    parser.add_argument(
        "--exclude-epochs",
        type=int,
        nargs="*",
        default=[85],
        help="Epoch numbers to ignore (defaults to [85] per assignment instructions).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testing"),
        help="Directory where report and plots will be written.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="testing_report.md",
        help="Markdown filename for the generated report.",
    )
    parser.add_argument(
        "--tracker-prefix",
        type=str,
        default="seqtrack_b256_",
        help="Prefix used for per-epoch directories under --results-root.",
    )
    parser.add_argument(
        "--precision-threshold",
        type=float,
        default=20.0,
        help="Pixel threshold to compute precision (default 20).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sequence progress information.",
    )
    return parser.parse_args()


def list_epoch_directories(
    results_root: Path,
    tracker_prefix: str,
    include_epochs: Optional[Sequence[int]],
    exclude_epochs: Sequence[int],
) -> List[Tuple[int, Path]]:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    exclude_set = set(exclude_epochs or [])

    epoch_dirs: List[Tuple[int, Path]] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(tracker_prefix):
            continue
        try:
            epoch_num = int(name.split("_")[-1])
        except ValueError:
            continue
        if include_epochs is not None and epoch_num not in include_epochs:
            continue
        if epoch_num in exclude_set:
            continue
        epoch_dirs.append((epoch_num, child))

    if not epoch_dirs:
        raise RuntimeError("No epoch result folders found matching the specified filters.")

    epoch_dirs.sort(key=lambda item: item[0])
    return epoch_dirs


def load_boxes(path: Path) -> np.ndarray:
    data = np.loadtxt(str(path), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, 4)
    return data


def load_visibility(sequence_root: Path, length: int) -> Optional[np.ndarray]:
    occ_path = sequence_root / "full_occlusion.txt"
    oov_path = sequence_root / "out_of_view.txt"
    if not occ_path.exists() or not oov_path.exists():
        return None
    occ = np.loadtxt(str(occ_path), delimiter=",")
    oov = np.loadtxt(str(oov_path), delimiter=",")
    occ = np.asarray(occ).reshape(-1)
    oov = np.asarray(oov).reshape(-1)
    vis = np.logical_and(occ == 0, oov == 0)
    if vis.shape[0] >= length:
        return vis[:length].astype(bool)
    padded = np.ones(length, dtype=bool)
    padded[: vis.shape[0]] = vis.astype(bool)
    return padded


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    top_left = np.maximum(pred[:, :2], gt[:, :2])
    bottom_right = np.minimum(pred[:, :2] + pred[:, 2:] - 1.0, gt[:, :2] + gt[:, 2:] - 1.0)
    wh = np.clip(bottom_right - top_left + 1.0, a_min=0.0, a_max=None)
    intersection = wh[:, 0] * wh[:, 1]

    pred_area = np.clip(pred[:, 2], 0.0, None) * np.clip(pred[:, 3], 0.0, None)
    gt_area = np.clip(gt[:, 2], 0.0, None) * np.clip(gt[:, 3], 0.0, None)
    union = pred_area + gt_area - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def compute_center_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_center = pred[:, :2] + 0.5 * (pred[:, 2:] - 1.0)
    gt_center = gt[:, :2] + 0.5 * (gt[:, 2:] - 1.0)
    diff = pred_center - gt_center
    return np.sqrt((diff ** 2).sum(axis=1))


def ensure_lengths(pred: np.ndarray, gt: np.ndarray, visibility: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    length = min(len(pred), len(gt))
    pred = pred[:length]
    gt = gt[:length]
    vis = visibility[:length] if visibility is not None else None
    return pred, gt, vis


def compute_sequence_stats(
    sequence_name: str,
    pred_path: Path,
    dataset_root: Path,
    precision_threshold: float,
) -> Optional[SequenceStats]:
    class_name = sequence_name.split("-")[0]
    seq_root = dataset_root / class_name / sequence_name
    gt_path = seq_root / "groundtruth.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth for {sequence_name}: {gt_path}")

    pred_boxes = load_boxes(pred_path)
    gt_boxes = np.loadtxt(str(gt_path), delimiter=",", dtype=np.float64)
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes.reshape(1, 4)

    visibility = load_visibility(seq_root, gt_boxes.shape[0])
    pred_boxes, gt_boxes, visibility = ensure_lengths(pred_boxes, gt_boxes, visibility)

    valid_mask = (gt_boxes[:, 2] > 0.0) & (gt_boxes[:, 3] > 0.0)
    if visibility is not None:
        valid_mask &= visibility

    if not np.any(valid_mask):
        return None

    iou = compute_iou(pred_boxes, gt_boxes)
    center_error = compute_center_error(pred_boxes, gt_boxes)

    iou_valid = iou[valid_mask]
    center_valid = center_error[valid_mask]

    iou_mean = float(iou_valid.mean())
    precision_at_20 = float(np.mean(center_valid <= precision_threshold))

    success_curve = np.array([np.mean(iou_valid >= thr) for thr in IOU_THRESHOLDS])
    precision_curve = np.array([np.mean(center_valid <= thr) for thr in PRECISION_THRESHOLDS])
    auc = float(success_curve.mean())

    time_path = pred_path.with_name(f"{sequence_name}_time.txt")
    total_time = float(np.sum(np.loadtxt(str(time_path)))) if time_path.exists() else 0.0

    return SequenceStats(
        sequence=sequence_name,
        class_name=class_name,
        num_frames=len(pred_boxes),
        valid_frames=int(valid_mask.sum()),
        iou_curve=success_curve,
        precision_curve=precision_curve,
        iou_mean=iou_mean,
        precision_at_20=precision_at_20,
        auc=auc,
        total_time_sec=total_time,
    )


def evaluate_epoch(
    epoch: int,
    epoch_dir: Path,
    dataset_root: Path,
    precision_threshold: float,
    verbose: bool = False,
) -> EpochStats:
    sequence_stats: List[SequenceStats] = []
    for pred_file in sorted(epoch_dir.glob("lasot/*.txt")):
        if pred_file.name.endswith("_time.txt"):
            continue
        sequence_name = pred_file.stem
        if verbose:
            print(f"[epoch {epoch}] computing metrics for {sequence_name}...")
        stats = compute_sequence_stats(sequence_name, pred_file, dataset_root, precision_threshold)
        if stats is None:
            continue
        sequence_stats.append(stats)

    if not sequence_stats:
        raise RuntimeError(f"No sequences evaluated for epoch {epoch} (check dataset path).")

    iou_curve = np.mean([seq.iou_curve for seq in sequence_stats], axis=0)
    precision_curve = np.mean([seq.precision_curve for seq in sequence_stats], axis=0)

    iou_mean = float(np.mean([seq.iou_mean for seq in sequence_stats]) * 100.0)
    precision_mean = float(np.mean([seq.precision_at_20 for seq in sequence_stats]) * 100.0)
    auc_mean = float(iou_curve.mean() * 100.0)

    total_frames = int(np.sum([seq.valid_frames for seq in sequence_stats]))
    total_time = float(np.sum([seq.total_time_sec for seq in sequence_stats]))
    fps = total_frames / total_time if total_time > 0 else 0.0
    ms_per_frame = (total_time / total_frames * 1000.0) if total_frames > 0 else 0.0

    return EpochStats(
        epoch=epoch,
        num_sequences=len(sequence_stats),
        total_frames=total_frames,
        total_time_sec=total_time,
        iou_mean=iou_mean,
        precision_mean=precision_mean,
        auc_mean=auc_mean,
        fps=fps,
        ms_per_frame=ms_per_frame,
    )


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def plot_metrics(
    epoch_stats: Sequence[EpochStats],
    output_dir: Path,
) -> Dict[str, Path]:
    epochs = [stat.epoch for stat in epoch_stats]
    iou = [stat.iou_mean for stat in epoch_stats]
    precision = [stat.precision_mean for stat in epoch_stats]
    auc = [stat.auc_mean for stat in epoch_stats]
    fps = [stat.fps for stat in epoch_stats]

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "testing_metrics.png"
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, iou, marker="o", label="IoU (%)")
    plt.plot(epochs, precision, marker="s", label="Precision (%)")
    plt.plot(epochs, auc, marker="^", label="AUC (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title("Evaluation Metrics vs Epoch")
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()

    auc_path = output_dir / "testing_auc.png"
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, auc, marker="^", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("AUC (%)")
    plt.title("AUC vs Epoch")
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(auc_path, dpi=300)
    plt.close()

    fps_path = output_dir / "testing_fps.png"
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, fps, marker="o", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Frames per Second")
    plt.title("Inference FPS vs Epoch")
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(fps_path, dpi=300)
    plt.close()

    return {"summary": summary_path, "auc": auc_path, "fps": fps_path}


def write_report(
    epoch_stats: Sequence[EpochStats],
    plot_paths: Dict[str, Path],
    output_dir: Path,
    report_name: str,
) -> Path:
    report_lines: List[str] = []
    report_lines.append("# SeqTrack Testing Report")
    report_lines.append("")
    report_lines.append(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    report_lines.append("")

    report_lines.append("## Inference Metrics")
    inference_rows = [
        [f"{stat.epoch}", f"{stat.fps:.2f}", f"{stat.ms_per_frame:.2f}"]
        for stat in epoch_stats
    ]
    report_lines.append("")
    report_lines.append("### Table 1: Inference Rate")
    report_lines.append(format_table(["Epoch", "FPS", "ms/frame"], inference_rows))
    report_lines.append("")

    report_lines.append("## Evaluation Metrics")
    evaluation_rows = [
        [
            f"{stat.epoch}",
            f"{stat.iou_mean:.2f}",
            f"{stat.precision_mean:.2f}",
            f"{stat.auc_mean:.2f}",
        ]
        for stat in epoch_stats
    ]
    report_lines.append("")
    report_lines.append("### Table 2: Evaluation Results")
    report_lines.append(format_table(["Epoch", "IoU (%)", "Precision (%)", "AUC (%)"], evaluation_rows))
    report_lines.append("")

    report_lines.append("## Evaluation Graphs")
    report_lines.append(f"![Overall Metrics]({plot_paths['summary'].name})")
    report_lines.append(f"![AUC vs Epoch]({plot_paths['auc'].name})")
    report_lines.append(f"![Inference FPS vs Epoch]({plot_paths['fps'].name})")
    report_lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / report_name
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()

    dataset_root: Optional[Path] = args.dataset_root
    if dataset_root is None:
        env_override = os.environ.get("LASOT_ROOT") or os.environ.get("LASOT_PATH")
        if env_override:
            dataset_root = Path(env_override)
        else:
            env_default = Path.cwd() / "datasets" / "LaSOT"
            if env_default.exists():
                dataset_root = env_default
            else:
                raise RuntimeError(
                    "Dataset root not specified. Provide --dataset-root pointing to the LaSOTBenchmark/train folder."
                )
    dataset_root = dataset_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    epoch_dirs = list_epoch_directories(
        results_root=args.results_root.expanduser().resolve(),
        tracker_prefix=args.tracker_prefix,
        include_epochs=args.epochs,
        exclude_epochs=args.exclude_epochs or [],
    )

    precision_threshold = args.precision_threshold

    epoch_stats: List[EpochStats] = []
    for epoch, path in epoch_dirs:
        stats = evaluate_epoch(
            epoch=epoch,
            epoch_dir=path,
            dataset_root=dataset_root,
            precision_threshold=precision_threshold,
            verbose=args.verbose,
        )
        epoch_stats.append(stats)
        print(
            f"[epoch {epoch}] IoU={stats.iou_mean:.2f}%  Precision@{precision_threshold:.0f}px={stats.precision_mean:.2f}%  "
            f"AUC={stats.auc_mean:.2f}%  FPS={stats.fps:.2f}"
        )

    plot_paths = plot_metrics(epoch_stats, args.output_dir.expanduser().resolve())
    report_path = write_report(epoch_stats, plot_paths, args.output_dir.expanduser().resolve(), args.report_name)

    print(f"\nReport written to {report_path}")
    for label, path in plot_paths.items():
        print(f"Saved {label} plot to {path}")


if __name__ == "__main__":
    main()


