# Kaggle Training Setup with HuggingFace Auto-Upload

## Quick Start Instructions

### 1. Clone and Setup Repository

```python
%cd /kaggle/working
!rm -rf Assignment_5
!git clone https://github.com/Amrzxk/Assignment_5.git
%cd Assignment_5/SeqTrack

!apt-get update && apt-get install -y libturbojpeg
!pip install --upgrade --force-reinstall numpy==1.26.4 tensorboard==2.14.0 tensorboardX==2.6
!pip install -r requirements.txt
```

### 2. Download LaSOT Dataset (10 Classes)

```python
from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile, shutil

classes = ["electricfan","elephant","flag","fox","frog","gametarget","gecko","giraffe","goldfish","gorilla"]
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
print("✓ Dataset ready!")
```

### 3. Setup Local Environment Config

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
```

### 4. Update Config File

```python
%%writefile lib/config/seqtrack/config.py
from easydict import EasyDict as edict
import yaml

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.BINS = 4000
cfg.MODEL.FEATURE_TYPE = 'x'

cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = 'vit_base_patch16'
cfg.MODEL.ENCODER.DROP_PATH = 0.1
cfg.MODEL.ENCODER.PRETRAIN_TYPE = 'mae'
cfg.MODEL.ENCODER.STRIDE = 16
cfg.MODEL.ENCODER.USE_CHECKPOINT = True

cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.NHEADS = 8
cfg.MODEL.DECODER.DROPOUT = 0.1
cfg.MODEL.DECODER.DIM_FEEDFORWARD = 1024
cfg.MODEL.DECODER.DEC_LAYERS = 2
cfg.MODEL.DECODER.PRE_NORM = False
cfg.MODEL.DECODER.HIDDEN_DIM = 256
cfg.MODEL.DECODER.BINS = 4000
cfg.MODEL.DECODER.FEATURE_TYPE = 'x'

cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.LR_BACKBONE = 0.00001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 100
cfg.TRAIN.LR_DROP_EPOCH = 80
cfg.TRAIN.BATCH_SIZE = 64
cfg.TRAIN.NUM_WORKER = 4
cfg.TRAIN.OPTIMIZER = 'ADAMW'
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1
cfg.TRAIN.FREEZE_ENCODER = False
cfg.TRAIN.ENCODER_OPEN = []
cfg.TRAIN.CE_WEIGHT = 1.0
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.GRAD_CLIP_NORM = 0.5
cfg.TRAIN.GRAD_ACCUM_STEPS = 1
cfg.TRAIN.AMP = False
cfg.TRAIN.WARMUP_EPOCHS = 5
cfg.TRAIN.LOSS_WEIGHTS = edict({'bbox': 2.0, 'giou': 2.0, 'ce': 1.0})
cfg.TRAIN.CHECKPOINT_INTERVAL = 5
cfg.TRAIN.KEEP_LAST_CHECKPOINT_EPOCHS = 0
cfg.TRAIN.SAVE_EVERY_EPOCH = False
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = 'step'
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.5
cfg.TRAIN.SCHEDULER.MILESTONES = []
cfg.TRAIN.SCHEDULER.GAMMA = 0.1

cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = 'sequence'
cfg.DATA.LOADER = 'tracking'
cfg.DATA.SEQ_FORMAT = 'xywh'

cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5

cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 2
cfg.DATA.TEMPLATE.SIZE = 256
cfg.DATA.TEMPLATE.FACTOR = 4.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0.0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0.0

cfg.DATA.LASOT = edict()
cfg.DATA.LASOT.CLASS_SELECTION_PATH = 'class_selection.txt'
cfg.DATA.LASOT.ROOT = None

cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ['LASOT']
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 5000
cfg.DATA.TRAIN.CLASSES_FILE = 'class_selection.txt'
cfg.DATA.TRAIN.MAX_SAMPLE_PER_SEQ = 50
cfg.DATA.TRAIN.FREEZE_ENCODER = True
cfg.DATA.TRAIN.ENCODER_OPEN = ['blocks.10', 'blocks.11']
cfg.DATA.TRAIN.PRINT_INTERVAL = 50
cfg.DATA.TRAIN.SAVE_INTERVAL = 10

cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 4.0
cfg.TEST.TEMPLATE_SIZE = 256
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 100
cfg.TEST.WINDOW = False
cfg.TEST.NUM_TEMPLATES = 1
cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.DEFAULT = 9999
cfg.TEST.UPDATE_INTERVALS.LASOT = 450
cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = 1
cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = 25
cfg.TEST.UPDATE_INTERVALS.VOT20 = 10
cfg.TEST.UPDATE_INTERVALS.VOT21 = 10
cfg.TEST.UPDATE_INTERVALS.VOT22 = 10
cfg.TEST.UPDATE_THRESHOLD = edict()
cfg.TEST.UPDATE_THRESHOLD.DEFAULT = 0.6
cfg.TEST.UPDATE_THRESHOLD.VOT20 = 0.475
cfg.TEST.UPDATE_THRESHOLD.VOT21 = 0.475
cfg.TEST.UPDATE_THRESHOLD.VOT22 = 0.475

def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return

def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return

def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
```

### 5. Set HuggingFace Token & Launch Training

**IMPORTANT:** Get your HF token from https://huggingface.co/settings/tokens and replace `YOUR_HF_TOKEN_HERE` below.

```python
%%bash
cd /kaggle/working/Assignment_5/SeqTrack

# Set HuggingFace credentials for auto-upload
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export HF_REPO="hossamaladdin/Assignment5"
export HF_FOLDER="Member 3"

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory
mkdir -p /kaggle/working/output

# Launch training with live console output
python -u lib/train/run_training.py \
    --script seqtrack \
    --config seqtrack_b256 \
    --save_dir /kaggle/working/output \
    --use_lmdb 0 | tee /kaggle/working/output/train_stream.log
```

## What Happens During Training

1. **Every 5 epochs**, a checkpoint is saved locally
2. **Immediately after saving**, the checkpoint is uploaded to:
   - Repo: `hossamaladdin/Assignment5`
   - Path: `Member 3/checkpoints/SeqTrackEpoch00XX.pth.tar`
3. **After upload**, the local checkpoint is **deleted** to free disk space
4. You can monitor uploads in the console logs (look for `✓ Uploaded` messages)

## Checkpoint Naming

- Epoch 5: `SeqTrackEpoch0005.pth.tar`
- Epoch 10: `SeqTrackEpoch0010.pth.tar`
- ...
- Epoch 100: `SeqTrackEpoch0100.pth.tar`

Total: **20 checkpoints** will be uploaded to HuggingFace.

## View Checkpoints Online

Visit: [https://huggingface.co/datasets/hossamaladdin/Assignment5/tree/main/Member%203](https://huggingface.co/datasets/hossamaladdin/Assignment5/tree/main/Member%203)

## Troubleshooting

### If upload fails:
- Check HF token is valid
- Ensure you have write access to the dataset repo
- Check Kaggle internet connection

### If disk fills up:
Run manual cleanup:
```python
!rm -rf /kaggle/working/output/checkpoints/train/seqtrack/seqtrack_b256/*.pth.tar
```

### Monitor disk usage:
```python
!df -h /kaggle/working
```

