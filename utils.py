import torch
import numpy as np
from scipy.stats import spearmanr
from torchmetrics.regression import SpearmanCorrCoef
import io
import os
import math
import time
import json
import builtins
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
#from torch import inf
inf = float('inf')
import random
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor
from timm.data.transforms import str_to_interp_mode
from itertools import chain
from scipy.spatial.transform import Rotation as R
# from tensorboardX import SummaryWriter
import wandb
from PIL import Image
import cv2
from scipy.ndimage.filters import gaussian_filter


try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    xm = xmp = pl = xu = None


XLA_CFG = {"is_xla": False, "logging_interval": 20}

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if XLA_CFG["is_xla"]:
            t = torch.tensor([self.count, self.total], device=xm.xla_device())
            t = xm.all_reduce(xm.REDUCE_SUM, t).tolist()
            self.count = int(t[0])
            self.total = t[1]
            return
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')
        self.data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if not XLA_CFG["is_xla"] and torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            self.data_time.update(time.time() - end)
            yield obj
            self.iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = self.iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if not XLA_CFG["is_xla"] and torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandBLogger(object):
    def __init__(self, log_dir, args=None, project="multiviewMAE"):
        self.args = args
        import tempfile
        self.writer = wandb.init(project=project, 
                            entity="peisen_zhou",
                            dir=tempfile.gettempdir(),
                            config=args)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
        # self.writer.log({}, commit=True)

    def update(self, head='scalar', step=None, commit=False, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            if "frames" in k:
                # import pdb; pdb.set_trace()
                v = torch.cat(v)
                img_grid = make_grid(v, nrow=self.args.num_frames, normalize=not True, scale_each=not True)
                img_grid = F.to_pil_image(img_grid.clip(0,0.996))
                # img_grid = Image.fromarray((img_grid.movedim(0,2).cpu().numpy() * 255).astype(np.uint8))
                img_grid = wandb.Image(img_grid, caption=f"input _ pred _ label")
                self.writer.log({k: [img_grid]}, step = self.step if step is None else step, commit=commit)
                continue
            elif "maps" in k:
                v = torch.cat(v, 0)
                v = torch.moveaxis(v, 3, 1)
                img_grid = make_grid(v, nrow=1, normalize=True, scale_each=False)
                img_grid = F.to_pil_image(img_grid.clip(0, 0.996))
                img_grid = wandb.Image(img_grid)
                self.writer.log({k: [img_grid]}, step = self.step if step is None else step, commit=commit)
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.log({head + "/" + k: v}, step = self.step if step is None else step, commit=commit)

    def flush(self):
        # self.writer.flush()
        pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if args == ('torch_xla.core.xla_model::mark_step',):
            # XLA server step tracking
            if is_master:
                builtin_print(*args, **kwargs)
            return
        force = force or (not XLA_CFG["is_xla"] and get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if XLA_CFG["is_xla"]:
        raise Exception("This function should not be called in XLA")
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if XLA_CFG["is_xla"]:
        return xm.xrt_world_size()
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if XLA_CFG["is_xla"]:
        return xm.get_ordinal()
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if XLA_CFG["is_xla"]:
        xm.save(*args, **kwargs, global_master=True)
        return
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    print("Init distributed mode")
    if XLA_CFG["is_xla"]:
        args.rank = xm.get_ordinal()
        args.distributed = True
        setup_for_distributed(args.rank == 0)
        return
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        print('dist_on_itp')
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = int(os.environ['SLURM_LOCALID'])
    #     args.world_size = int(os.environ['SLURM_NTASKS'])
    #     os.environ['RANK'] = str(args.rank)
    #     os.environ['LOCAL_RANK'] = str(args.gpu)
    #     os.environ['WORLD_SIZE'] = str(args.world_size)

    #     node_list = os.environ['SLURM_NODELIST']
    #     addr = subprocess.getoutput(
    #         f'scontrol show hostname {node_list} | head -n1')
    #     if 'MASTER_ADDR' not in os.environ:
    #         os.environ['MASTER_ADDR'] = addr
    #     print('SLURM_PROCID')
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print('RANK/WORLD_SIZE')
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)


def broadcast_xla_master_model_param(model, args):
    """
    Broadcast the model parameters from master process to other processes
    """
    parameters_and_buffers = []
    is_master = xm.is_master_ordinal(local=False)
    for p in chain(model.parameters(), model.buffers()):
        # Set all params in non-master devices to zero so that all_reduce is
        # equivalent to broadcasting parameters from master to other devices.
        scale = 1 if is_master else 0
        scale = torch.tensor(scale, dtype=p.data.dtype, device=p.data.device)
        p.data.mul_(scale)
        parameters_and_buffers.append(p.data)
    xm.wait_device_ops()
    xm.all_reduce(xm.REDUCE_SUM, parameters_and_buffers)
    xm.mark_step()
    xm.rendezvous("broadcast_xla_master_model_param")

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    print("akash: lenSched", len(schedule), "epochs:", epochs, "niter_per_ep:", niter_per_ep)
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, linear_model_without_ddp=None, linear_optimizer=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if linear_model_without_ddp is not None:
        linear_weight = linear_model_without_ddp.state_dict()
        linear_optimizer_state = linear_optimizer.state_dict()
    if args.lora_layers is not None:
        if args.lora_attn != "mlp":
            lora_merged = model_without_ddp.encoder.blocks[-1].attn.qkv.merged
        else:
            lora_merged = model_without_ddp.encoder.blocks[-1].mlp.fc1.merged
    else:
        lora_merged = None
    if loss_scaler != None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'linear_model': linear_weight,
                'optimizer': optimizer.state_dict(),
                'linear_optimizer': linear_optimizer_state,
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'lora_merged': lora_merged,
                'args': args,
            }

            if model_ema != None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema != None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, linear_model_without_ddp=None, linear_optimizer=None):
    output_dir = Path(args.output_dir)
    if loss_scaler != None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            if linear_model_without_ddp is not None and 'linear_model' in checkpoint.keys():
                linear_model_without_ddp.load_state_dict(checkpoint['linear_model'])
            if linear_optimizer is not None and 'lienar_optimizer' in checkpoint.keys():
                linear_optimizer.load_state_dict(checkpoint['linear_optimizer'])
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'lora_merged' in checkpoint:
                if args.lora_layers is not None:
                    if args.lora_layers == 'all':
                        lora_layers = list(range(len(model_without_ddp.encoder.blocks)))
                    elif args.lora_layers == 'last':
                        lora_layers = [-1]
                    else:
                        lora_layers = [int(item) for item in args.lora_layers.split(',')]
                    for i in lora_layers:
                        if args.lora_attn == "mlp":
                            model_without_ddp.encoder.blocks[i].mlp.fc1.merged = checkpoint['lora_merged']
                            model_without_ddp.encoder.blocks[i].mlp.fc2.merged = checkpoint['lora_merged']
                        else:
                            model_without_ddp.encoder.blocks[i].attn.qkv.merged = checkpoint['lora_merged']
                    print(f"Updated LoRA to merged:{checkpoint['lora_merged']}")
    else:  
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        if XLA_CFG["is_xla"]:
            x_reduce = torch.tensor(x, device=xm.xla_device())
            return xm.all_reduce(xm.REDUCE_SUM, x_reduce, scale=1.0 / world_size).item()
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def matrix_to_quaternion(matrix):
    r = R.from_matrix(matrix)
    return r.as_quat()

def matrix_to_euler(matrix):
    r = R.from_matrix(matrix)
    return r.as_euler("XYZ")

def euler_to_matrix(euler):
    r = R.from_euler("xyz", euler)
    return r.as_matrix()

def euler_to_quaternion(euler):
    r = R.from_euler("xyz", euler)
    return r.as_quat()

def quaternion_to_matrix(quat):
    r = R.from_quat(quat)
    return r.as_matrix()

def quaternion_to_euler(quat):
    r = R.from_quat(quat)
    return r.as_euler("xyz")

def camera_to_category(rot_mat, translation):
    euler = matrix_to_euler(rot_mat)
    euler_class =  np.any([np.all([euler>0, euler<=math.pi],axis=0), np.all([euler<=0, euler<=-math.pi], axis=0)], axis=0).astype(int)
    translation_class = (translation>0).astype(int)
    euler_class = 4*euler_class[0] + 2*euler_class[1] + euler_class[2]
    translation_class = 4*translation_class[0] + 2*translation_class[1] + translation_class[2]
    combined_class = euler_class + 8*translation_class
    return combined_class

def get_transform_center_crop(data_config):
    input_size = data_config['input_size']
    if isinstance(input_size, (tuple, list)):
        input_size = input_size[-2:]
    else:
        input_size = (input_size, input_size)
    mean = data_config['mean']
    std = data_config['std']
    tf = []
    tf += [transforms.CenterCrop(224)]
    tf += [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return transforms.Compose(tf)

def get_transform_wo_crop(data_config):
    input_size = data_config['input_size']
    if isinstance(input_size, (tuple, list)):
        input_size = input_size[-2:]
    else:
        input_size = (input_size, input_size)
    mean = data_config['mean']
    std = data_config['std']
    interpolation = data_config['interpolation']
    tf = []
    tf += [transforms.Resize(input_size[0], interpolation=str_to_interp_mode(interpolation))]

    tf += [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return transforms.Compose(tf)

def gaussian_kernel(size=10, sigma=10):
    """
    Generates a 2D Gaussian kernel.

    Parameters
    ----------
    size : int, optional
        Kernel size, by default 10
    sigma : int, optional
        Kernel sigma, by default 10

    Returns
    -------
    kernel : torch.Tensor
        A Gaussian kernel.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return kernel

def gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.

    Parameters
    ----------
    heatmap : torch.Tensor
        The heatmap to blur.
    kernel : torch.Tensor
        The Gaussian kernel.

    Returns
    -------
    blurred_heatmap : torch.Tensor
        The blurred heatmap.
    """
    # Ensure heatmap and kernel have the correct dimensions
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = torch.nn.functional.conv2d(heatmap, kernel, padding='same')

    return blurred_heatmap[0]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def create_label_index_map(anchor_file):
        '''
        Create a single label to index map to stay consistent across data splits
        '''
        label_to_index = {}
        index = 0
        with open(anchor_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.split('/')[1]
                if label not in label_to_index:
                    label_to_index[label] = index
                    index += 1
        return label_to_index

def save_as_overlay(img, mask, filename, percentile=99, save=True):
    #span = abs(np.percentile(mask, percentile))
    #vmin = -span
    #vmax = span
    vmax = np.max(mask)
    vmin = np.min(mask)
    #mask = np.clip((mask-vmin)/(vmax-vmin), 0, 1)
    mask = (mask-vmin)/(vmax-vmin)
    alpha = np.ones((mask.shape[0], mask.shape[1]))
    thresh = abs(np.percentile(mask, 90))
    alpha[np.logical_and(mask>=0, mask<thresh)] = 0
    alpha[np.logical_and(mask<0, mask>-thresh)] = 0
    #alpha[mask==0] = 0
    mask = (mask+1)/2
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2RGBA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    overlay = cv2.addWeighted(heatmap, 0.6, img, 0.4, 0)
    img[alpha!=0] = overlay[alpha!=0]
    if save:
        cv2.imwrite(filename, img)
    return img

def get_cvm_attn_mask(q_len, num_frames):
    q_len = q_len // num_frames
    attn_mask = torch.tril(torch.ones((num_frames, num_frames))).to(torch.bool)
    attn_mask = attn_mask.repeat_interleave(q_len, dim=1, output_size=q_len*num_frames).repeat_interleave(q_len, dim=0, output_size=q_len*num_frames)
    return attn_mask
        

def create_clickmap(point_lists, image_shape, exponential_decay=False, tau=0.5):
    """
    Create a clickmap from click points.

    Args:
        click_points (list of tuples): List of (x, y) coordinates where clicks occurred.
        image_shape (tuple): Shape of the image (height, width).
        blur_kernel (torch.Tensor, optional): Gaussian kernel for blurring. Default is None.
        tau (float, optional): Decay rate for exponential decay. Default is 0.5 but this needs to be tuned.

    Returns:
        np.ndarray: A 2D array representing the clickmap, blurred if kernel provided.
    """
    heatmap = np.zeros(image_shape, dtype=np.uint8)
    for click_points in point_lists:
        if exponential_decay:
            for idx, point in enumerate(click_points):

                if 0 <= point[1] < image_shape[0] and 0 <= point[0] < image_shape[1]:
                    heatmap[point[1], point[0]] += np.exp(-idx / tau)
        else:
            for point in click_points:
                if 0 <= point[1] < image_shape[0] and 0 <= point[0] < image_shape[1]:
                    heatmap[point[1], point[0]] += 1
    return heatmap


def spearman_correlation_np(heatmaps_a: np.ndarray, heatmaps_b: np.ndarray) -> np.ndarray:
    """
    Computes the Spearman correlation between two sets of heatmaps using NumPy.

    Parameters
    ----------
    heatmaps_a : np.ndarray
        First set of heatmaps. Expected shape (N, W, H).
    heatmaps_b : np.ndarray
        Second set of heatmaps. Expected shape (N, W, H).

    Returns
    -------
    np.ndarray
        Array of Spearman correlation scores between the two sets of heatmaps.

    Raises
    ------
    AssertionError
        If the shapes of the input heatmaps are not identical or not (N, W, H).
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    heatmaps_a = _ensure_numpy(heatmaps_a)
    heatmaps_b = _ensure_numpy(heatmaps_b)

    return np.array([spearmanr(ha.flatten(), hb.flatten())[0] for ha, hb in zip(heatmaps_a, heatmaps_b)])

def spearman_correlation(heatmaps_a: torch.Tensor, heatmaps_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the Spearman correlation between two sets of heatmaps using PyTorch.

    Parameters
    ----------
    heatmaps_a : torch.Tensor
        First set of heatmaps. Expected shape (N, W, H).
    heatmaps_b : torch.Tensor
        Second set of heatmaps. Expected shape (N, W, H).

    Returns
    -------
    torch.Tensor
        Tensor of Spearman correlation scores between the two sets of heatmaps.

    Raises
    ------
    AssertionError
        If the shapes of the input heatmaps are not identical or not (N, W, H).
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    batch_size = heatmaps_a.shape[0]
    spearman = SpearmanCorrCoef(num_outputs=1)
    heatmaps_a = heatmaps_a.reshape(batch_size, -1)
    heatmaps_b = heatmaps_b.reshape(batch_size, -1)

    return torch.stack([spearman(heatmaps_a[i], heatmaps_b[i]) for i in range(batch_size)])

def compute_human_alignment(predicted_heatmaps: torch.Tensor, clickme_heatmaps: torch.Tensor) -> float:
    """
    Computes the human alignment score between predicted heatmaps and ClickMe heatmaps.

    Parameters
    ----------
    predicted_heatmaps : torch.Tensor
        Predicted heatmaps. Expected shape (N, C, W, H) or (N, W, H).
    clickme_heatmaps : torch.Tensor
        ClickMe heatmaps. Expected shape (N, C, W, H) or (N, W, H).

    Returns
    -------
    float
        Human alignment score.
    """
    HUMAN_SPEARMAN_CEILING = 0.65753

    predicted_heatmaps = _ensure_3d(predicted_heatmaps)
    clickme_heatmaps = _ensure_3d(clickme_heatmaps)

    scores = spearman_correlation(predicted_heatmaps, clickme_heatmaps)
    human_alignment = scores.mean().item() / HUMAN_SPEARMAN_CEILING

    return human_alignment

def _ensure_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Ensures the input tensor is a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to convert.

    Returns
    -------
    np.ndarray
        NumPy array version of the input tensor.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.cpu().numpy()

def _ensure_3d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Ensures the input heatmaps are 3D by removing the channel dimension if present.

    Parameters
    ----------
    heatmaps : torch.Tensor
        Input heatmaps. Expected shape (N, C, W, H) or (N, W, H).

    Returns
    -------
    torch.Tensor
        3D heatmaps with shape (N, W, H).
    """
    if len(heatmaps.shape) == 4:
        return heatmaps[:, 0, :, :]
    return heatmaps

def gaussian_kernel(size=10, sigma=10):
    """
    Generates a 2D Gaussian kernel.

    Parameters
    ----------
    size : int, optional
        Kernel size, by default 10
    sigma : int, optional
        Kernel sigma, by default 10

    Returns
    -------
    kernel : torch.Tensor
        A Gaussian kernel.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return kernel

def gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.

    Parameters
    ----------
    heatmap : torch.Tensor
        The heatmap to blur.
    kernel : torch.Tensor
        The Gaussian kernel.

    Returns
    -------
    blurred_heatmap : torch.Tensor
        The blurred heatmap.
    """
    # Ensure heatmap and kernel have the correct dimensions
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = torch.nn.functional.conv2d(heatmap, kernel, padding='same')

    return blurred_heatmap[0]