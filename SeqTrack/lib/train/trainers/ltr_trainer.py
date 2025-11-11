import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import lib.utils.misc as misc


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        accum_steps = max(1, getattr(self.settings, "grad_accum_steps", 1))

        if loader.training:
            self.optimizer.zero_grad(set_to_none=True)

        for i, data in enumerate(loader, 1):
            # print("start")
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                if not self.use_amp:
                    loss = loss / accum_steps
                    loss.backward()
                    do_step = (i % accum_steps == 0) or (i == loader.__len__())
                    if do_step:
                        if self.settings.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.scaler.scale(loss / accum_steps).backward()
                    do_step = (i % accum_steps == 0) or (i == loader.__len__())
                    if do_step:
                        if self.settings.grad_clip_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)


    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            world_size = max(1, getattr(self.settings, "world_size", 1))
            local_rank = getattr(self.settings, "local_rank", -1)
            rank_display = local_rank if local_rank not in [-1, None] else 0
            rank_label = f"Rank {rank_display}"

            local_total = loader.__len__()
            global_total = local_total * world_size
            if world_size > 1:
                global_batch_idx = (i - 1) * world_size + rank_display + 1
            else:
                global_batch_idx = i

            global_pct = (global_batch_idx / global_total) * 100 if global_total else 0
            local_pct = (i / local_total) * 100 if local_total else 0

            batch_size_local = getattr(self.settings, "batchsize", batch_size)
            samples_per_epoch_cfg = getattr(self.settings, "samples_per_epoch", None)
            default_samples_target = global_total * batch_size_local
            samples_target = samples_per_epoch_cfg if samples_per_epoch_cfg is not None else default_samples_target
            samples_target = int(samples_target)
            samples_processed_global = min(global_batch_idx * batch_size_local, samples_target)
            samples_pct = (samples_processed_global / samples_target) * 100 if samples_target else 0

            print_str = (f"[{rank_label}] Epoch {self.epoch}/{self.settings.num_epochs} | "
                         f"Global Batch {global_batch_idx}/{global_total} ({global_pct:.1f}%) | "
                         f"Local Batch {i}/{local_total} ({local_pct:.1f}%) | "
                         f"Samples {samples_processed_global}/{samples_target} ({samples_pct:.1f}%) | "
                         f"FPS avg {average_fps:.1f} (batch {batch_fps:.1f})")

            current_lr = None
            if self.lr_scheduler is not None:
                try:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                except Exception:
                    try:
                        current_lr = self.optimizer.param_groups[0]['lr']
                    except Exception:
                        current_lr = None
            if current_lr is not None:
                print_str += f" | LR {current_lr:.2e}"

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    if 'loss' in name.lower():
                        print_str += f" | {name}: {val.avg:.4f}"
                    elif 'iou' in name.lower():
                        print_str += f" | {name}: {val.avg:.3f}"
                    else:
                        print_str += f" | {name}: {val.avg:.5f}"

            print(print_str, flush=True)

            if misc.is_main_process():
                with open(self.settings.log_file, 'a') as f:
                    f.write(print_str + '\n')
                    f.flush()

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    if hasattr(self.lr_scheduler, 'get_last_lr'):
                        lr_list = self.lr_scheduler.get_last_lr()
                    else:
                        lr_list = self.lr_scheduler.get_lr()
                except Exception:
                    try:
                        lr_list = self.lr_scheduler._get_lr(self.epoch)
                    except AttributeError:
                        lr_list = []
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
