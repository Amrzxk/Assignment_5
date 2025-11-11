import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.seqtrack import build_seqtrack
# forward propagation related
from lib.train.actors import SeqTrackActor
# for import modules
import importlib


def _apply_encoder_freeze(net, cfg):
    train_cfg = getattr(cfg.DATA, "TRAIN", {})
    if not train_cfg.get("FREEZE_ENCODER", False):
        return
    open_layers = train_cfg.get("ENCODER_OPEN", []) or []
    if hasattr(net, "encoder"):
        for name, param in net.encoder.named_parameters():
            param.requires_grad = False
        for layer in open_layers:
            for name, param in net.encoder.named_parameters():
                if layer in name:
                    param.requires_grad = True
    if hasattr(net, "module") and hasattr(net.module, "encoder"):
        _apply_encoder_freeze(net.module, cfg)


def run(settings):
    settings.description = 'Training script for SeqTrack'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file) #update cfg from experiments
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")


    # Create network
    if settings.script_name == "seqtrack":
        net = build_seqtrack(cfg)        #pix2seq method with multi-frames and encoder mask
    else:
        raise ValueError("illegal script name")

    # Potentially freeze encoder layers before wrapping in DDP
    _apply_encoder_freeze(net, cfg)

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "seqtrack":
        bins = cfg.MODEL.BINS
        weight = torch.ones(bins + 2)
        weight[bins] = 0.01
        weight[bins + 1] = 0.01
        objective = {'ce': CrossEntropyLoss(weight=weight)}
        ce_weight = settings.loss_weights.get('ce', settings.loss_weights.get('bbox', cfg.TRAIN.CE_WEIGHT if hasattr(cfg.TRAIN, 'CE_WEIGHT') else 1.0))
        loss_weight = {'ce': ce_weight}
        actor = SeqTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # Setup HuggingFace checkpoint uploader
    if settings.local_rank in [-1, 0]:  # Only upload from main process
        try:
            from lib.train.admin.hf_uploader import HFCheckpointUploader
            # Get HF token from environment variable
            hf_token = os.environ.get("HF_TOKEN", None)
            hf_repo = os.environ.get("HF_REPO", "hossamaladdin/Assignment5")
            hf_folder = os.environ.get("HF_FOLDER", "Member 3")
            
            if hf_token:
                trainer.hf_uploader = HFCheckpointUploader(
                    repo_id=hf_repo,
                    token=hf_token,
                    folder_path=hf_folder
                )
                print("\n✓ HuggingFace auto-upload enabled")
                print(f"  Checkpoints will be uploaded to: {hf_repo}/{hf_folder}/checkpoints/\n")
            else:
                print("⚠ HF_TOKEN not set, checkpoint auto-upload disabled")
                trainer.hf_uploader = None
        except Exception as e:
            print(f"⚠ HF uploader setup failed: {e}")
            trainer.hf_uploader = None

    # Store total epochs for better progress display
    settings.num_epochs = cfg.TRAIN.EPOCH
    
    # Print training start banner
    if settings.local_rank in [-1, 0]:
        print("\n" + "="*80)
        print("🎯 TRAINING CONFIGURATION")
        print("="*80)
        print(f"  Total Epochs: {cfg.TRAIN.EPOCH}")
        print(f"  Batch Size: {cfg.TRAIN.BATCH_SIZE}")
        print(f"  Learning Rate: {cfg.TRAIN.LR:.2e}")
        print(f"  LR Backbone: {cfg.TRAIN.LR_BACKBONE:.2e}")
        print(f"  Gradient Accumulation Steps: {getattr(cfg.TRAIN, 'GRAD_ACCUM_STEPS', 1)}")
        print(f"  Mixed Precision (AMP): {getattr(cfg.TRAIN, 'AMP', False)}")
        print(f"  Checkpoint Interval: Every {getattr(cfg.TRAIN, 'CHECKPOINT_INTERVAL', 10)} epochs")
        print(f"  Dataset: {cfg.DATA.TRAIN.DATASETS_NAME}")
        print(f"  Classes: {getattr(cfg.DATA.TRAIN, 'CLASSES_FILE', 'All')}")
        print(f"  Samples Per Epoch: {cfg.DATA.TRAIN.SAMPLE_PER_EPOCH}")
        print("="*80 + "\n")
    
    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
