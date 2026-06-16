import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import RandomErasing
from timm.data import RandAugment

class DataConfig:
    DATA_DIR = "/root/autodl-tmp/imagenet100"
    BATCH_SIZE = 384
    INPUT_SIZE = 224
    NUM_WORKERS = 8
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            DataConfig.INPUT_SIZE,
            scale=(0.08, 1.0),
            ratio=(3.0/4.0, 4.0/3.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(DataConfig.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform

def get_dataloaders():
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DataConfig.DATA_DIR, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DataConfig.DATA_DIR, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DataConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=DataConfig.NUM_WORKERS,
        pin_memory=DataConfig.PIN_MEMORY,
        drop_last=True,
        prefetch_factor=DataConfig.PREFETCH_FACTOR,
        persistent_workers=True if DataConfig.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DataConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=DataConfig.NUM_WORKERS,
        pin_memory=DataConfig.PIN_MEMORY,
        persistent_workers=True if DataConfig.NUM_WORKERS > 0 else False
    )

    return train_loader, val_loader

if __name__ == "__main__":
    if not os.path.exists(DataConfig.DATA_DIR):
        raise FileNotFoundError(f"数据集路径不存在：{DataConfig.DATA_DIR}")

    train_loader, val_loader = get_dataloaders()
    print(f"输入尺寸: {DataConfig.INPUT_SIZE}×{DataConfig.INPUT_SIZE}）")
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"训练集批次数量: {len(train_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        print(f"批次图像维度: {images.shape} (设备: {images.device})")
        print(f"批次标签维度: {labels.shape} (设备: {labels.device})")
        break