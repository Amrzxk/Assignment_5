import torch
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LinearLR, SequentialLR
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, Imagenet1k
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': getattr(cfg.DATA.TEMPLATE, "FACTOR", None),
                                   'search': getattr(cfg.DATA.SEARCH, "FACTOR", None)}
    settings.output_sz = {'template': getattr(cfg.DATA.TEMPLATE, "SIZE", 128),
                          'search': getattr(cfg.DATA.SEARCH, "SIZE", 256)}
    settings.center_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "CENTER_JITTER", None),
                                     'search':getattr(cfg.DATA.SEARCH, "CENTER_JITTER", None)}
    settings.scale_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "SCALE_JITTER", None),
                                    'search': getattr(cfg.DATA.SEARCH, "SCALE_JITTER", None)}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE

    train_cfg = getattr(cfg.DATA, "TRAIN", {})
    lasot_cfg = getattr(cfg.DATA, "LASOT", {})

    settings.samples_per_epoch = getattr(train_cfg, "SAMPLE_PER_EPOCH", None)
    settings.lasot_class_selection_path = getattr(lasot_cfg, "CLASS_SELECTION_PATH", None)
    settings.lasot_root_override = getattr(lasot_cfg, "ROOT", None)

    classes_file = getattr(train_cfg, "CLASSES_FILE", None)
    if classes_file:
        settings.lasot_class_selection_path = classes_file
    settings.max_sample_per_seq = getattr(train_cfg, "MAX_SAMPLE_PER_SEQ", None)
    settings.freeze_encoder_cfg = getattr(train_cfg, "FREEZE_ENCODER", False)
    settings.encoder_open_layers = getattr(train_cfg, "ENCODER_OPEN", [])
    settings.extra_save_interval = getattr(train_cfg, "SAVE_INTERVAL", None)

    settings.checkpoint_interval = getattr(cfg.TRAIN, "CHECKPOINT_INTERVAL", 10)
    if settings.extra_save_interval:
        settings.checkpoint_interval = settings.extra_save_interval
    settings.keep_last_checkpoint_epochs = getattr(cfg.TRAIN, "KEEP_LAST_CHECKPOINT_EPOCHS", 10)
    settings.save_every_epoch = getattr(cfg.TRAIN, "SAVE_EVERY_EPOCH", False)
    settings.grad_accum_steps = getattr(cfg.TRAIN, "GRAD_ACCUM_STEPS", 1)
    settings.warmup_epochs = getattr(cfg.TRAIN, "WARMUP_EPOCHS", 0)
    loss_weights_cfg = getattr(cfg.TRAIN, "LOSS_WEIGHTS", None)
    if loss_weights_cfg is None:
        settings.loss_weights = {'ce': getattr(cfg.TRAIN, "CE_WEIGHT", 1.0)}
    else:
        settings.loss_weights = dict(loss_weights_cfg)
        if 'ce' not in settings.loss_weights:
            settings.loss_weights['ce'] = settings.loss_weights.get('bbox', getattr(cfg.TRAIN, "CE_WEIGHT", 1.0))


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    lasot_class_selection = getattr(settings, "lasot_class_selection_path", None)
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID", "TRACKINGNET", "IMAGENET1K"]
        if name == "LASOT":
            lasot_root = getattr(settings.env, "lasot_dir", None)
            if not lasot_root:
                lasot_root = getattr(settings, "lasot_root_override", None)
            if not lasot_root:
                raise ValueError("LaSOT root directory is not configured. "
                                 "Set it in lib/train/admin/local.py or via DATA.LASOT.ROOT in the config.")
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(lasot_root, split='train', image_loader=image_loader,
                                      class_selection_path=lasot_class_selection))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "IMAGENET1K":
            datasets.append(Imagenet1k(settings.env.imagenet1k_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SeqTrackProcessing(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint,
                                                          settings=settings)

    # Train sampler and loader
    sampler_mode_cfg = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    sampler_mode = sampler_mode_cfg
    if sampler_mode_cfg.lower() in ["sequence", "order"]:
        sampler_mode = "order"
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode
                                            )

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    return loader_train


def get_optimizer_scheduler(net, cfg):
    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if "encoder" in n and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
        },
    ]
    if is_main_process():
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")

    warmup_epochs = getattr(cfg.TRAIN, "WARMUP_EPOCHS", 0)
    schedulers = []
    milestones = []

    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH,
                                                         gamma=getattr(cfg.TRAIN.SCHEDULER, "DECAY_RATE", 0.1))
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                              gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")

    if warmup_epochs and warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
        schedulers.extend([warmup, main_scheduler])
        milestones.append(warmup_epochs)
        lr_scheduler = SequentialLR(optimizer, schedulers, milestones)
    else:
        lr_scheduler = main_scheduler

    return optimizer, lr_scheduler
