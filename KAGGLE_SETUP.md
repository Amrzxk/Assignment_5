# Kaggle Evaluation Setup (SeqTrack Assignment 5)

This guide walks you through reproducing the testing pipeline for both Member 3 class groups on Kaggle.  
You will download the prepared LaSOT subset, pull checkpoints from Hugging Face, run evaluation for each model, and generate comparison plots/tables.

---

## 1. Clone the repository and install dependencies

Run the following cells in a new Kaggle Notebook (GPU → T4 x2 recommended):

```python
%cd /kaggle/working
!rm -rf Assignment_5
!git clone https://github.com/Amrzxk/Assignment_5.git
%cd Assignment_5/SeqTrack

!apt-get update && apt-get install -y libturbojpeg
!pip install --upgrade --force-reinstall numpy==1.26.4 tensorboard==2.14.0 tensorboardX==2.6
!pip install -r requirements.txt huggingface-hub
```

---

## 2. Download the 10 LaSOT classes (groups 21–30)

```python
from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile, shutil

classes = [
    "electricfan","elephant","flag","fox","frog",
    "gametarget","gecko","giraffe","goldfish","gorilla"
]

root = Path("/kaggle/working/LaSOT")
train_root = root / "LaSOTBenchmark" / "train"
train_root.mkdir(parents=True, exist_ok=True)

for cls in classes:
    print(f"Downloading {cls}...")
    zip_path = Path(hf_hub_download("l-lt/LaSOT", f"{cls}.zip", repo_type="dataset", cache_dir="/kaggle/temp"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink()

    cls_dir = train_root / cls
    cls_dir.mkdir(exist_ok=True)
    for seq_dir in root.glob(f"{cls}-*"):
        shutil.move(str(seq_dir), cls_dir / seq_dir.name)

(root / "class_selection.txt").write_text("\n".join(classes))
shutil.copy(root / "class_selection.txt", train_root / "class_selection.txt")
print("✓ LaSOT subset ready!")
```

> **Tip:** Expect ~10 GB download. The script deletes each ZIP immediately to stay within Kaggle storage limits.

---

## 3. Configure environment paths

### 3.1 Training/admin paths (keeps training scripts happy)

```python
%%writefile lib/train/admin/local.py
class EnvironmentSettings:
    def __init__(self):
        base = "/kaggle/working/output"
        self.workspace_dir = base
        self.tensorboard_dir = base + "/tensorboard"
        self.pretrained_networks = base + "/pretrained_networks"
        self.lasot_dir = "/kaggle/working/LaSOT/LaSOTBenchmark/train"
        self.got10k_dir = ''
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = ''
        self.coco_lmdb_dir = ''
        self.imagenet1k_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
```

### 3.2 Evaluation/test paths (required for `evaluate_checkpoints.py`)

```python
%%writefile lib/test/evaluation/local.py
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    base = "/kaggle/working/output"
    settings.prj_dir = "/kaggle/working/Assignment_5/SeqTrack"
    settings.save_dir = base
    settings.results_path = base + "/testing/results"
    settings.segmentation_path = base + "/testing/segmentation"
    settings.network_path = base + "/testing/networks"
    settings.result_plot_path = base + "/testing/result_plots"

    settings.lasot_path = "/kaggle/working/LaSOT/LaSOTBenchmark/train"
    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.otb_path = ''

    return settings
```

---

## 4. Provide the Hugging Face credentials

Set the token (replace with the provided secret if different) and default repo/folder values.  
Member 3 checkpoints live in `Member 3/checkpoints`, Member 3.2 checkpoints in `Member 3.2/checkpoints`.

```python
import os

os.environ["HF_TOKEN"] = ""
os.environ["HF_REPO"] = "hossamaladdin/Assignment5"
```

> **Security note:** For public notebooks, add the token as a Kaggle secret instead of hard‑coding it.

---

## 5. Evaluate Model 1 (classes 21–25, Hugging Face folder `Member 3`)

This command downloads each checkpoint on demand, runs LaSOT evaluation for epochs 5–100 (step 5), and writes JSON/log files under `/kaggle/working/output/testing`.

```python
%%bash
cd /kaggle/working/Assignment_5/SeqTrack

python -u evaluate_checkpoints.py \
  --tracker_name seqtrack \
  --tracker_param seqtrack_b256 \
  --dataset_name lasot \
  --start_epoch 5 \
  --end_epoch 100 \
  --threads 0 \
  --num_gpus 1 \
  --data_root /kaggle/working/LaSOT/LaSOTBenchmark/train \
  --hf_repo "${HF_REPO}" \
  --hf_folder "Member 3" \
  --hf_subdir "checkpoints" \
  --output_name member3 \
  --hf_cache_dir /kaggle/temp/hf_cache
```

Key outputs:
- JSON metrics: `/kaggle/working/output/testing/member3_evaluation_results.json`
- Inference logs: `/kaggle/working/output/testing/inference_logs/member3_inference_log.txt`
- Tracker results: `/kaggle/working/output/testing/results/seqtrack/seqtrack_b256_*`

---

## 6. Evaluate Model 2 (classes 26–30, Hugging Face folder `Member 3.2`)

```python
%%bash
cd /kaggle/working/Assignment_5/SeqTrack

python -u evaluate_checkpoints.py \
  --tracker_name seqtrack \
  --tracker_param seqtrack_b256 \
  --dataset_name lasot \
  --start_epoch 5 \
  --end_epoch 100 \
  --threads 0 \
  --num_gpus 1 \
  --data_root /kaggle/working/LaSOT/LaSOTBenchmark/train \
  --hf_repo "${HF_REPO}" \
  --hf_folder "Member 3.2" \
  --hf_subdir "checkpoints" \
  --output_name member3_2 \
  --hf_cache_dir /kaggle/temp/hf_cache
```

Outputs mirror the previous step, using the `member3_2` prefix.

---

## 7. Generate per-model reports (tables + plots)

Run once per JSON (the script names outputs using `--output_prefix`).

```python
%%bash
cd /kaggle/working/Assignment_5/SeqTrack

python generate_report.py \
  --input member3_evaluation_results.json \
  --output_dir testing/member3_reports \
  --output_prefix member3

python generate_report.py \
  --input member3_2_evaluation_results.json \
  --output_dir testing/member3_2_reports \
  --output_prefix member3_2
```

Important artefacts (per model):
- Markdown report: `.../member3_report.md`
- Overall metrics plot: `.../member3_metrics.png`
- AUC / Precision / FPS plots
- Class-wise breakdowns (if available)

---

## 8. Compare both models side-by-side

The comparison script overlays IoU, Precision, AUC, and FPS curves, and writes a combined Markdown table.

```python
%%bash
cd /kaggle/working/Assignment_5/SeqTrack

python compare_evaluations.py \
  --inputs member3_evaluation_results.json member3_2_evaluation_results.json \
  --labels "Member 3" "Member 3.2" \
  --output_dir testing/comparisons \
  --output_prefix member3_vs_member3_2
```

Outputs:
- `member3_vs_member3_2_iou_comparison.png`
- `member3_vs_member3_2_precision_comparison.png`
- `member3_vs_member3_2_auc_comparison.png`
- `member3_vs_member3_2_fps_comparison.png`
- `member3_vs_member3_2_comparison_table.md`
- `member3_vs_member3_2_summary.txt`

---

## 9. Collect evidence for submission

Recommended artefacts (all under `/kaggle/working/output/testing`):

| Purpose | File(s) |
| --- | --- |
| Raw metrics + checkpoint paths | `member3_evaluation_results.json`, `member3_2_evaluation_results.json` |
| Console-style inference logs | `inference_logs/member3_inference_log.txt`, `inference_logs/member3_2_inference_log.txt` |
| Per-model plots & tables | `member3_reports/*`, `member3_2_reports/*` |
| Cross-model comparison | `comparisons/member3_vs_member3_2_*` |
| Tracker outputs (optional verification) | `results/seqtrack/seqtrack_b256_*` |

Use Kaggle’s file browser or `zip` command to download everything:

```python
!cd /kaggle/working/output && zip -r evaluation_artifacts.zip testing
```

---

## 10. Troubleshooting & Tips

- **Download errors / disk full:** Re-run the dataset cell; ensure `/kaggle/temp` has space. Delete unused folders with `!rm -rf /kaggle/working/output/tracking_results`.
- **HF authentication failure:** Verify `os.environ["HF_TOKEN"]` inside a cell.
- **Slow evaluation:** Each checkpoint evaluates the full LaSOT subset. Expect several hours for 20 checkpoints per model. Consider running in two sessions if necessary.
- **Resume evaluations:** Use `--start_epoch` / `--end_epoch` to limit the range (e.g., `--start_epoch 55` to re-run later checkpoints).
- **Reuse downloads:** Hugging Face files cache to `/kaggle/temp/hf_cache`; keep the notebook kernel alive to avoid re-downloading.

You now have tables, graphs, and logs comparing both class-group models at every 5th epoch checkpoint. Capture the key plots and Markdown tables for your assignment report. Good luck!


