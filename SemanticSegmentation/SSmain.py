import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import warnings

from SemanticSegmentation.FPN import FPNSegHead

warnings.filterwarnings("ignore")

PRETRAINED_WEIGHT = "/home/maxverstappen/projects/ReGeForceNet/best_EGCVMamba_xxx.pth"
NUM_CLASSES_CLS = 100
NUM_CLASSES_SEG = 151
DROP_PATH_RATE = 0.05
BACKBONE_OUT_CHANNELS = 384
INPUT_SIZE = (512, 512)

CHECKPOINT_DIR = ""
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
EPOCHS = 50
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
FP16 = torch.cuda.is_available()
RESUME = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(PRETRAINED_WEIGHT):
    exit(1)

from SemanticSegmentation.SSdata import create_dataloaders


def compute_miou(preds, masks, num_classes=151, ignore_index=255):
    mask = masks != ignore_index
    preds = preds[mask]
    masks = masks[mask]

    iou_per_class = []
    for cls in range(num_classes):
        true_cls = (masks == cls)
        pred_cls = (preds == cls)

        intersection = (true_cls & pred_cls).sum().float()
        union = (true_cls | pred_cls).sum().float()

        if union == 0:
            iou_per_class.append(torch.tensor(1.0).to(DEVICE))
        else:
            iou_per_class.append(intersection / union)

    miou = torch.stack(iou_per_class).mean()
    return miou.item() * 100


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS} Train")

    for step, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.int64)

        with autocast(enabled=FP16):
            outputs = model(imgs)
            assert outputs.shape[2:] == masks.shape[1:]
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), avg_loss=total_loss/(step+1))
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    pbar = tqdm(loader, desc="Validation")

    for imgs, masks in pbar:
        imgs = imgs.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.int64)

        with autocast(enabled=FP16):
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        batch_miou = compute_miou(preds, masks, num_classes=NUM_CLASSES_SEG)
        total_miou += batch_miou

        avg_miou = total_miou / (pbar.n + 1)
        pbar.set_postfix(val_loss=loss.item(), mIoU=avg_miou)

    avg_loss = total_loss / len(loader)
    final_miou = total_miou / len(loader)
    return avg_loss, final_miou

def main():
    BEST_MIOU = 0.0
    start_epoch = 0

    train_loader, val_loader = create_dataloaders()

    model = FPNSegHead(
        pretrained_weight_path=PRETRAINED_WEIGHT,
        input_size=INPUT_SIZE
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY
    )
    scaler = GradScaler(enabled=FP16)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS
    )

    if RESUME and os.path.exists(os.path.join(CHECKPOINT_DIR, "last_ckpt.pth")):
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "last_ckpt.pth"), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        BEST_MIOU = ckpt["best_miou"]
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        val_loss, val_miou = validate(model, val_loader, criterion)

        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        if epoch == 10:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(),
                lr=BASE_LR / 10,
                weight_decay=WEIGHT_DECAY
            )

        if val_miou > BEST_MIOU:
            BEST_MIOU = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_miou": BEST_MIOU,
                "optimizer_state_dict": optimizer.state_dict()
            }, os.path.join(CHECKPOINT_DIR, "best_ckpt.pth"))

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_miou": BEST_MIOU,
            "optimizer_state_dict": optimizer.state_dict()
        }, os.path.join(CHECKPOINT_DIR, "last_ckpt.pth"))

def test_inference():
    model = FPNSegHead(
        pretrained_weight_path=PRETRAINED_WEIGHT,
        input_size=INPUT_SIZE
    ).to(DEVICE)
    model.eval()

    dummy_input = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(DEVICE, dtype=torch.float32)

    with torch.no_grad():
        with autocast(enabled=FP16):
            output = model(dummy_input)

    assert output.shape[2:] == dummy_input.shape[2:]

if __name__ == "__main__":
    test_inference()
    main()