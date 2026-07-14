import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from .utils import SmoothedValue, accuracy


def _classification_autocast(device, precision):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _reduce_meters(device, *meters):
    values = []
    for meter in meters:
        values.extend([meter.total, meter.count])
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    result = []
    for index in range(0, len(values), 2):
        result.append((tensor[index] / tensor[index + 1].clamp_min(1)).item())
    return result


def train_classification_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    smoothing=0.1,
    clip_grad=5.0,
    ema=None,
    mixup_fn=None,
    criterion=None,
    precision="bf16",
    channels_last=True,
    show_progress=True,
):
    model.train()
    criterion = criterion or nn.CrossEntropyLoss(label_smoothing=smoothing)
    loss_meter = SmoothedValue()
    acc_meter = SmoothedValue()
    for images, targets in tqdm(loader, desc="train", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        hard_targets = targets
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        optimizer.zero_grad(set_to_none=True)
        with _classification_autocast(device, precision):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        acc1 = accuracy(outputs.detach(), hard_targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
    loss, acc1 = _reduce_meters(device, loss_meter, acc_meter)
    return {"loss": loss, "acc1": acc1}


@torch.no_grad()
def evaluate_classification(
    model,
    loader,
    device,
    precision="bf16",
    channels_last=True,
    show_progress=True,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_meter = SmoothedValue()
    acc1_meter = SmoothedValue()
    acc5_meter = SmoothedValue()
    for images, targets in tqdm(loader, desc="eval", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with _classification_autocast(device, precision):
            outputs = model(images)
            loss = criterion(outputs, targets)
        top5 = min(5, outputs.shape[1])
        acc1, acc5 = accuracy(outputs, targets, topk=(1, top5))
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))
    loss, acc1, acc5 = _reduce_meters(device, loss_meter, acc1_meter, acc5_meter)
    return {"loss": loss, "acc1": acc1, "acc5": acc5}


def train_segmentation_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    ignore_index=255,
    precision="bf16",
    channels_last=True,
    clip_grad=5.0,
    show_progress=True,
):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    meter = SmoothedValue()
    for images, masks in tqdm(loader, desc="train", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with _classification_autocast(device, precision):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), images.size(0))
    loss = _reduce_meters(device, meter)[0]
    return {"loss": loss}


@torch.no_grad()
def evaluate_segmentation(
    model,
    loader,
    device,
    num_classes=150,
    ignore_index=255,
    precision="bf16",
    channels_last=True,
    show_progress=True,
):
    model.eval()
    hist = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)
    for images, masks in tqdm(loader, desc="eval", leave=False, disable=not show_progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with _classification_autocast(device, precision):
            logits = model(images)
        preds = logits.argmax(dim=1)
        valid = masks != ignore_index
        encoded = num_classes * masks[valid] + preds[valid]
        hist += torch.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(hist, op=dist.ReduceOp.SUM)
    iou = hist.diag() / (hist.sum(1) + hist.sum(0) - hist.diag()).clamp_min(1)
    return {"miou": iou.mean().mul(100).item()}


def make_scaler(device, precision="fp16"):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
