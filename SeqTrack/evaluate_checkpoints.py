
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from tracking.test import run_tracker
from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.tracker import Tracker
from torch.serialization import add_safe_globals
from huggingface_hub.errors import EntryNotFoundError

try:
    from lib.train.admin.stats import AverageMeter, StatValue
    add_safe_globals([AverageMeter, StatValue])
except (ImportError, AttributeError):
    pass

try:
    from lib.train.admin.settings import Settings as TrainSettings
    add_safe_globals([TrainSettings])
except (ImportError, AttributeError):
    pass

try:
    from lib.train.admin.local import EnvironmentSettings
    add_safe_globals([EnvironmentSettings])
except (ImportError, AttributeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SeqTrack checkpoints across epochs")
    parser.add_argument("--tracker_name", default="seqtrack", help="Tracker name to evaluate")
    parser.add_argument("--tracker_param", default="seqtrack_b256", help="Tracker parameter name / config")
    parser.add_argument("--dataset_name", default="lasot", help="Evaluation dataset name")
    parser.add_argument("--start_epoch", type=int, default=1, help="First epoch to evaluate (inclusive)")
    parser.add_argument("--end_epoch", type=int, default=10, help="Last epoch to evaluate (inclusive)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to expose to the evaluation")
    parser.add_argument("--threads", type=int, default=0, help="Number of worker processes (0 = sequential)")
    parser.add_argument("--skip_inference", action="store_true", help="Only aggregate metrics from existing results")
    parser.add_argument("--force", action="store_true", help="Re-run evaluation and overwrite cached results")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Optional explicit directory containing SEQTRACK_epXXXX.pth.tar checkpoints",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional override for dataset root (defaults to env settings)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output (only epoch summaries)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom suffix for result files (defaults to tracker_param).",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="Hugging Face dataset repository id (e.g., 'hossamaladdin/Assignment5').",
    )
    parser.add_argument(
        "--hf_folder",
        type=str,
        default=None,
        help="Top-level folder inside the dataset repo (e.g., 'Member 3').",
    )
    parser.add_argument(
        "--hf_subdir",
        type=str,
        default="checkpoints",
        help="Subdirectory inside the folder containing checkpoints (default: 'checkpoints').",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face access token. Falls back to --hf_token_env variable.",
    )
    parser.add_argument(
        "--hf_token_env",
        type=str,
        default="HF_TOKEN",
        help="Environment variable name from which to read the Hugging Face token.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional custom cache directory for Hugging Face downloads.",
    )
    parser.add_argument(
        "--epoch_step",
        type=int,
        default=1,
        help="Stride between evaluated epochs (default 1). For checkpoints saved every N epochs, set this to N.",
    )
    parser.add_argument(
        "--test_num_templates",
        type=str,
        default="auto",
        help="Override cfg.TEST.NUM_TEMPLATES during evaluation ('auto' matches DATA.TEMPLATE.NUMBER).",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "evaluation"


class HFCheckpointManager:
    def __init__(
        self,
        repo_id: str,
        folder: str,
        subdir: str = "checkpoints",
        token: Optional[str] = None,
        token_env: str = "HF_TOKEN",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.repo_id = repo_id
        self.folder = folder.strip("/ ")
        self.subdir = subdir.strip("/ ")
        self.cache_dir = cache_dir
        self.token_env = token_env
        self.token = token or os.environ.get(token_env)
        self._hf_download = None

        if not self.repo_id or not self.folder:
            raise ValueError("Both repo_id and folder must be specified for Hugging Face downloads.")

    def _require_download_fn(self):
        if self._hf_download is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "huggingface_hub is required to download checkpoints. "
                    "Install it with `pip install huggingface-hub`."
                ) from exc
            self._hf_download = hf_hub_download

    def ensure(self, epoch: int) -> Path:
        if self.token is None:
            raise ValueError(
                "Hugging Face token is required but missing. "
                f"Provide it via --hf_token or set the environment variable {self.token_env}."
            )

        filename = f"SEQTRACK_ep{epoch:04d}.pth.tar"
        path_parts = [self.folder]
        if self.subdir:
            path_parts.append(self.subdir)
        path_parts.append(filename)
        repo_path = "/".join(path_parts)

        self._require_download_fn()
        downloaded_path = self._hf_download(
            repo_id=self.repo_id,
            filename=repo_path,
            repo_type="dataset",
            token=self.token,
            cache_dir=self.cache_dir,
        )
        return Path(downloaded_path)


def resolve_checkpoint(
    project_root: Path,
    save_dir: Path,
    param_name: str,
    epoch: int,
    override_dir: Path | None,
    hf_manager: Optional[HFCheckpointManager] = None,
) -> Path:
    """Locate the checkpoint file for the specified epoch."""

    candidates = [
        (override_dir / f"SEQTRACK_ep{epoch:04d}.pth.tar") if override_dir else None,
        project_root
        / "training_runs"
        / param_name
        / "checkpoints"
        / "train"
        / "seqtrack"
        / param_name
        / f"SEQTRACK_ep{epoch:04d}.pth.tar",
        project_root
        / "training_runs"
        / param_name
        / "checkpoints"
        / "train"
        / "seqtrack"
        / param_name
        / f"SEQTRACK_ep{epoch:04d}.pth.tar",
        save_dir
        / "training_runs"
        / param_name
        / "checkpoints"
        / "train"
        / "seqtrack"
        / param_name
        / f"SEQTRACK_ep{epoch:04d}.pth.tar",
        project_root
        / "training_runs"
        / param_name
        / f"epoch_{epoch}.ckpt",
        save_dir / "training_runs" / param_name / f"epoch_{epoch}.ckpt",
        save_dir / "checkpoints" / f"epoch_{epoch}.ckpt",
        save_dir
        / "checkpoints"
        / "train"
        / "seqtrack"
        / param_name
        / f"SEQTRACK_ep{epoch:04d}.pth.tar",
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.exists():
            return candidate

    if hf_manager is not None:
        downloaded = hf_manager.ensure(epoch)
        if downloaded.exists():
            return downloaded

    raise FileNotFoundError(
        f"Could not find checkpoint for epoch {epoch}. Checked: "
        + ", ".join(str(p) for p in candidates if p is not None)
    )


def collect_inference_stats(tracker: Tracker, dataset) -> Dict[str, float]:
    """Aggregate inference timing statistics from *_time.txt files."""

    total_frames = 0
    total_time = 0.0
    per_sequence = []

    for seq in dataset:
        timings_file = (
            Path(tracker.results_dir)
            / seq.dataset
            / f"{seq.name}_time.txt"
        )

        if not timings_file.exists():
            return {}

        times = np.loadtxt(timings_file, delimiter="\t")
        times = np.atleast_1d(times)

        seq_time = float(times.sum())
        seq_frames = int(times.shape[0])
        seq_fps = seq_frames / seq_time if seq_time > 0 else 0.0

        per_sequence.append(
            {
                "sequence": seq.name,
                "frames": seq_frames,
                "total_time_sec": seq_time,
                "fps": seq_fps,
            }
        )

        total_frames += seq_frames
        total_time += seq_time

    fps = total_frames / total_time if total_time > 0 else 0.0
    ms_per_frame = 1000.0 / fps if fps > 0 else 0.0

    return {
        "total_frames": total_frames,
        "total_time_sec": total_time,
        "fps": fps,
        "ms_per_frame": ms_per_frame,
        "per_sequence": per_sequence,
    }


def format_inference_log(epoch: int, stats: Dict[str, float], dataset_name: str) -> List[str]:
    """Generate human-readable log lines following the training log style."""

    if not stats:
        return []

    lines: List[str] = []
    for seq_stats in stats["per_sequence"]:
        line = (
            f"Epoch {epoch:02d} : Sequence {seq_stats['sequence']} , "
            f"frames : {seq_stats['frames']} , "
            f"total_time : {seq_stats['total_time_sec']:.4f} s , "
            f"FPS : {seq_stats['fps']:.3f}"
        )
        lines.append(line)

    summary = (
        f"Epoch {epoch:02d} : Dataset {dataset_name} , "
        f"frames : {stats['total_frames']} , "
        f"total_time : {stats['total_time_sec']:.4f} s , "
        f"FPS : {stats['fps']:.3f} , "
        f"ms_per_frame : {stats['ms_per_frame']:.3f}"
    )
    lines.append(summary)
    return lines


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def verify_results_exist(tracker: Tracker, dataset) -> List[Path]:
    """Return list of missing result files for a tracker/dataset combo."""

    missing: List[Path] = []
    base_dir = Path(tracker.results_dir)

    for seq in dataset:
        result_file = base_dir / seq.dataset / f"{seq.name}.txt"
        if not result_file.exists():
            missing.append(result_file)

    return missing


def get_auc_curve(ave_success_rate_plot_overlap: torch.Tensor, valid_sequence: torch.Tensor):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)
    return auc_curve, auc


def get_prec_curve(ave_success_rate_plot_center: torch.Tensor, valid_sequence: torch.Tensor):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, 20]
    return prec_curve, prec_score


def main() -> None:
    args = parse_args()

    if args.start_epoch > args.end_epoch:
        raise ValueError("start_epoch must be <= end_epoch")

    if args.epoch_step <= 0:
        raise ValueError("--epoch_step must be a positive integer")
    epochs = list(range(args.start_epoch, args.end_epoch + 1, args.epoch_step))
    if not epochs:
        raise ValueError("No epochs selected; check start/end/epoch_step arguments")
    output_slug = slugify(args.output_name or args.tracker_param)

    template_override = (args.test_num_templates or "").strip()
    if template_override:
        if template_override.lower() != "auto":
            try:
                override_int = int(template_override)
            except ValueError as exc:
                raise ValueError(
                    "--test_num_templates must be 'auto' or a positive integer"
                ) from exc
            if override_int <= 0:
                raise ValueError("--test_num_templates must be greater than zero when provided as an integer")
            template_override = str(override_int)
        os.environ["SEQTRACK_TEST_NUM_TEMPLATES"] = template_override

    settings = env_settings()
    project_root = Path(settings.prj_dir)
    save_dir = Path(settings.save_dir)
    testing_root = save_dir / "testing"
    ensure_dir(testing_root)

    result_plot_root = getattr(settings, "result_plot_path", "")
    if not result_plot_root:
        result_plot_root = str(testing_root / "result_plots")
    result_plot_path = Path(result_plot_root)
    ensure_dir(result_plot_path)

    inference_log_dir = Path(getattr(settings, "inference_log_path", testing_root / "inference_logs"))
    ensure_dir(inference_log_dir)

    override_checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    data_root_override = Path(args.data_root) if args.data_root else None
    hf_manager: Optional[HFCheckpointManager] = None

    if args.hf_repo and args.hf_folder:
        hf_manager = HFCheckpointManager(
            repo_id=args.hf_repo,
            folder=args.hf_folder,
            subdir=args.hf_subdir,
            token=args.hf_token,
            token_env=args.hf_token_env,
            cache_dir=args.hf_cache_dir,
        )

    # Apply dataset override if requested
    if data_root_override is not None:
        if args.dataset_name.lower() == "lasot":
            settings.lasot_path = str(data_root_override)
        else:
            raise ValueError("--data_root override currently supported only for LaSOT")

    dataset = get_dataset(args.dataset_name)

    trackers: List[Tracker] = []
    tracker_by_epoch: Dict[int, Tracker] = {}
    checkpoint_map: Dict[int, str] = {}
    inference_stats_map: Dict[int, Dict[str, float]] = {}
    inference_log_lines: List[str] = []
    evaluated_epochs: List[int] = []

    for epoch in epochs:
        try:
            checkpoint_path = resolve_checkpoint(
                project_root, save_dir, args.tracker_param, epoch, override_checkpoint_dir, hf_manager
            )
        except EntryNotFoundError:
            if not args.quiet:
                print(f"[skip] Checkpoint for epoch {epoch:04d} not found on Hugging Face; skipping.")
            continue
        checkpoint_map[epoch] = str(checkpoint_path)

        tracker = Tracker(args.tracker_name, args.tracker_param, args.dataset_name, run_id=epoch)
        trackers.append(tracker)
        tracker_by_epoch[epoch] = tracker

        results_dir = Path(tracker.results_dir)
        need_inference = not args.skip_inference

        if results_dir.exists():
            if args.force and not args.skip_inference:
                if not args.quiet:
                    print(f"[clean] Removing existing results for epoch {epoch}: {results_dir}")
                remove_dir(results_dir)
            else:
                if not args.skip_inference:
                    if not args.quiet:
                        print(
                            f"[skip] Results already exist for epoch {epoch} at {results_dir}. "
                            "Use --force to recompute."
                        )
                    need_inference = False

        if need_inference:
            if not args.quiet:
                print(
                    f"[run] Epoch {epoch}: running tracker '{args.tracker_name}' with params "
                    f"'{args.tracker_param}' on dataset '{args.dataset_name}'."
                )
            os.environ["CHECKPOINT_EPOCH"] = str(epoch)
            os.environ["CHECKPOINT_PATH"] = str(checkpoint_path)
            os.environ["SEQTRACK_DISABLE_LOGGING"] = "1" if args.quiet else "0"
            try:
                run_tracker(
                    args.tracker_name,
                    args.tracker_param,
                    run_id=epoch,
                    dataset_name=args.dataset_name,
                    debug=0,
                    threads=args.threads,
                    num_gpus=args.num_gpus,
                )
            finally:
                os.environ.pop("CHECKPOINT_EPOCH", None)
                os.environ.pop("CHECKPOINT_PATH", None)
                os.environ.pop("SEQTRACK_DISABLE_LOGGING", None)

        missing_files = verify_results_exist(tracker, dataset)
        if missing_files:
            if args.force:
                if not args.quiet:
                    print(f"[warn] Missing results for epoch {epoch}. Forcing re-run.")
                    for missing in missing_files:
                        print(f"    missing: {missing}")
                if results_dir.exists():
                    remove_dir(results_dir)

                os.environ["CHECKPOINT_EPOCH"] = str(epoch)
                os.environ["CHECKPOINT_PATH"] = str(checkpoint_path)
                os.environ["SEQTRACK_DISABLE_LOGGING"] = "1" if args.quiet else "0"
                try:
                    run_tracker(
                        args.tracker_name,
                        args.tracker_param,
                        run_id=epoch,
                        dataset_name=args.dataset_name,
                        debug=0,
                        threads=args.threads,
                        num_gpus=args.num_gpus,
                    )
                finally:
                    os.environ.pop("CHECKPOINT_EPOCH", None)
                    os.environ.pop("CHECKPOINT_PATH", None)
                    os.environ.pop("SEQTRACK_DISABLE_LOGGING", None)

                missing_files = verify_results_exist(tracker, dataset)
                if missing_files:
                    raise RuntimeError(
                        f"Evaluation for epoch {epoch} missing results even after force re-run: "
                        + ", ".join(str(p) for p in missing_files)
                    )
            else:
                if not args.quiet:
                    print(f"[warn] Missing results for epoch {epoch}. Skipping metrics. Use --force to re-run.")
                inference_stats_map[epoch] = {}
                if trackers:
                    trackers.pop()
                tracker_by_epoch.pop(epoch, None)
                continue

        stats = collect_inference_stats(tracker, dataset)
        inference_stats_map[epoch] = stats
        inference_log_lines.extend(format_inference_log(epoch, stats, args.dataset_name))
        evaluated_epochs.append(epoch)

    if not evaluated_epochs:
        print("No checkpoints evaluated. Exiting.")
        return

    report_name = f"{output_slug}_{args.dataset_name}"
    report_dir = result_plot_path / report_name
    if args.force:
        remove_dir(report_dir)

    eval_data = extract_results(trackers, dataset, report_name, skip_missing_seq=False)

    valid_sequence = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    avg_overlap_all = torch.tensor(eval_data["avg_overlap_all"], dtype=torch.float64) * 100.0
    mean_iou = avg_overlap_all[valid_sequence].mean(dim=0)

    auc_curve, auc_values = get_auc_curve(
        torch.tensor(eval_data["ave_success_rate_plot_overlap"], dtype=torch.float64),
        valid_sequence,
    )
    _, precision_values = get_prec_curve(
        torch.tensor(eval_data["ave_success_rate_plot_center"], dtype=torch.float64),
        valid_sequence,
    )

    metrics_by_epoch: Dict[int, Dict[str, float]] = {}
    for idx, tracker_info in enumerate(eval_data["trackers"]):
        run_id = tracker_info["run_id"]
        metrics_by_epoch[run_id] = {
            "IoU": float(mean_iou[idx].item()),
            "Precision": float(precision_values[idx].item()),
            "AUC": float(auc_values[idx].item()),
        }

    results_payload = {
        "tracker_name": args.tracker_name,
        "tracker_param": args.tracker_param,
        "dataset_name": args.dataset_name,
        "report_name": report_name,
        "output_name": output_slug,
        "epoch_range": {"start": args.start_epoch, "end": args.end_epoch, "step": args.epoch_step},
        "metadata": {
            "hf_repo": args.hf_repo,
            "hf_folder": args.hf_folder,
            "hf_subdir": args.hf_subdir if args.hf_repo else None,
            "checkpoint_source": "huggingface" if hf_manager else "local",
        },
        "epochs": [],
    }

    for epoch in evaluated_epochs:
        inference_stats = inference_stats_map.get(epoch, {})
        inference_entry = {
            "total_frames": int(inference_stats.get("total_frames", 0)),
            "total_time_sec": float(inference_stats.get("total_time_sec", 0.0)),
            "fps": float(inference_stats.get("fps", 0.0)),
            "ms_per_frame": float(inference_stats.get("ms_per_frame", 0.0)),
        }

        tracker_ref = tracker_by_epoch.get(epoch)
        results_dir_str = str(Path(tracker_ref.results_dir)) if tracker_ref is not None else ""

        epoch_entry = {
            "epoch": epoch,
            "checkpoint_path": checkpoint_map.get(epoch),
            "results_dir": results_dir_str,
            "metrics": metrics_by_epoch.get(epoch, {}),
            "inference": inference_entry,
        }
        results_payload["epochs"].append(epoch_entry)

    evaluation_json_path = testing_root / f"{output_slug}_evaluation_results.json"
    with evaluation_json_path.open("w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)

    inference_log_path = inference_log_dir / f"{output_slug}_inference_log.txt"
    with inference_log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("\n".join(inference_log_lines))

    print("\nEvaluation summary:")
    for entry in results_payload["epochs"]:
        epoch = entry["epoch"]
        metrics = entry["metrics"]
        inference = entry["inference"]
        print(
            f"  Epoch {epoch:02d} | IoU: {metrics.get('IoU', 0.0):5.2f}% | "
            f"Precision: {metrics.get('Precision', 0.0):5.2f}% | "
            f"AUC: {metrics.get('AUC', 0.0):5.2f}% | "
            f"FPS: {inference.get('fps', 0.0):6.2f}"
        )

    print(f"\nSaved evaluation results to {evaluation_json_path}")
    print(f"Inference log written to {inference_log_path}")
    print(f"Result plots stored in {report_dir}")


if __name__ == "__main__":
    main()


