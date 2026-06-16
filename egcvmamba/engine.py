import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .utils import SmoothedValue, accuracy


def train_classification_epoch(model, loader, optimizer, device, scaler, smoothing=0.1, clip_grad=5.0, ema=None):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    loss_meter = SmoothedValue()
    acc_meter = SmoothedValue()
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
    return {"loss": loss_meter.avg, "acc1": acc_meter.avg}


@torch.no_grad()
def evaluate_classification(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_meter = SmoothedValue()
    acc1_meter = SmoothedValue()
    acc5_meter = SmoothedValue()
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))
    return {"loss": loss_meter.avg, "acc1": acc1_meter.avg, "acc5": acc5_meter.avg}


def train_segmentation_epoch(model, loader, optimizer, device, scaler, ignore_index=255):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    meter = SmoothedValue()
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), images.size(0))
    return {"loss": meter.avg}


@torch.no_grad()
def evaluate_segmentation(model, loader, device, num_classes=150, ignore_index=255):
    model.eval()
    hist = torch.zeros(num_classes, num_classes, device=device)
    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        valid = masks != ignore_index
        encoded = num_classes * masks[valid] + preds[valid]
        hist += torch.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    iou = hist.diag() / (hist.sum(1) + hist.sum(0) - hist.diag()).clamp_min(1)
    return {"miou": iou.mean().mul(100).item()}


def make_scaler(device):
    return GradScaler(enabled=device.type == "cuda")
