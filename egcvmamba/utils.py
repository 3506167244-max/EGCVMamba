import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_mode():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process training requires CUDA GPUs.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier(device_ids=[local_rank])
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def parameter_groups(model, weight_decay, lr=None):
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exclude = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or name.endswith("A_log")
            or name.endswith(".D")
        )
        (no_decay if exclude else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if lr is not None:
        for group in groups:
            group["lr"] = float(lr)
    return groups


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    values = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        values.append(correct_k.mul_(100.0 / batch_size))
    return values


class SmoothedValue:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)


class EMA:
    def __init__(self, model, decay=0.9998):
        self.decay = float(decay)
        self.module = deepcopy(unwrap_model(model)).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = unwrap_model(model).state_dict()
        for name, value in self.module.state_dict().items():
            current = source[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                value.copy_(current)

    def state_dict(self):
        return {"decay": self.decay, "module": self.module.state_dict()}

    def load_state_dict(self, state_dict):
        self.decay = float(state_dict.get("decay", self.decay))
        if "module" in state_dict:
            self.module.load_state_dict(state_dict["module"])
            return

        # Compatibility with checkpoints written by the initial repository.
        shadow = state_dict.get("shadow", {})
        current = self.module.state_dict()
        current.update({name: value for name, value in shadow.items() if name in current})
        self.module.load_state_dict(current)

    @torch.no_grad()
    def copy_to(self, model):
        unwrap_model(model).load_state_dict(self.module.state_dict())


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
