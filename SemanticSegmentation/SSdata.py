import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

ADE_ROOT = "/home/maxverstappen/projects/ADEChallengeData2016"
TRAIN_BATCH_SIZE = 20
VAL_BATCH_SIZE = 10
NUM_WORKERS = 4
IMAGE_SIZE = 512
NUM_CLASSES = 151
IGNORE_INDEX = 255


def validate_ade_path(ade_root):
    if not os.path.exists(ade_root):
        raise FileNotFoundError(ade_root)

    img_train_path = os.path.join(ade_root, "images/training")
    img_val_path = os.path.join(ade_root, "images/validation")
    ann_train_path = os.path.join(ade_root, "annotations/training")
    ann_val_path = os.path.join(ade_root, "annotations/validation")

    required_folders = [img_train_path, img_val_path, ann_train_path, ann_val_path]
    for folder in required_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(folder)
    return ade_root


def get_transforms(is_train=True):
    if is_train:
        transform = A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})
    else:
        transform = A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})
    return transform


class ADESegDataset(Dataset):
    def __init__(self, ade_root, split="train", transform=None):
        self.ade_path = validate_ade_path(ade_root)
        self.split = split
        self.transform = transform

        if self.split == "train":
            self.img_dir = os.path.join(self.ade_path, "images/training")
            self.mask_dir = os.path.join(self.ade_path, "annotations/training")
        else:
            self.img_dir = os.path.join(self.ade_path, "images/validation")
            self.mask_dir = os.path.join(self.ade_path, "annotations/validation")

        self.img_names = [f.split(".")[0] for f in os.listdir(self.img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_name}.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = np.array(img, dtype=np.uint8)
        mask = np.array(mask, dtype=np.int64)

        mask[mask == IGNORE_INDEX] = 0

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
            mask = mask.to(dtype=torch.int64)

        assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert mask.dtype == torch.int64

        return img, mask


def create_dataloaders():
    validate_ade_path(ADE_ROOT)

    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)

    train_dataset = ADESegDataset(
        ade_root=ADE_ROOT,
        split="train",
        transform=train_transform
    )
    val_dataset = ADESegDataset(
        ade_root=ADE_ROOT,
        split="val",
        transform=val_transform
    )

    num_workers = NUM_WORKERS if os.name != "nt" else 0

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()