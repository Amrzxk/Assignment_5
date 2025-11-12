from easydict import EasyDict as edict
import yaml

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.BINS = 4000
cfg.MODEL.FEATURE_TYPE = 'x'

# MODEL.ENCODER
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = 'vit_base_patch16'
cfg.MODEL.ENCODER.DROP_PATH = 0.1
cfg.MODEL.ENCODER.PRETRAIN_TYPE = 'mae'
cfg.MODEL.ENCODER.STRIDE = 16
cfg.MODEL.ENCODER.USE_CHECKPOINT = True

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.NHEADS = 8
cfg.MODEL.DECODER.DROPOUT = 0.1
cfg.MODEL.DECODER.DIM_FEEDFORWARD = 1024
cfg.MODEL.DECODER.DEC_LAYERS = 2
cfg.MODEL.DECODER.PRE_NORM = False
cfg.MODEL.DECODER.HIDDEN_DIM = 256
cfg.MODEL.DECODER.BINS = 4000
cfg.MODEL.DECODER.FEATURE_TYPE = 'x'

# TRAIN
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

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = 'step'
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.5
cfg.TRAIN.SCHEDULER.MILESTONES = []
cfg.TRAIN.SCHEDULER.GAMMA = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = 'sequence'
cfg.DATA.LOADER = 'tracking'
cfg.DATA.SEQ_FORMAT = 'xywh'

# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5

# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 2
cfg.DATA.TEMPLATE.SIZE = 256
cfg.DATA.TEMPLATE.FACTOR = 4.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0.0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0.0

# DATA.LASOT (required by YAML)
cfg.DATA.LASOT = edict()
cfg.DATA.LASOT.CLASS_SELECTION_PATH = 'class_selection.txt'
cfg.DATA.LASOT.ROOT = None

# DATA.TRAIN (required by YAML)
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

# TEST
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