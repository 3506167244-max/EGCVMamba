from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def build_imagenet_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.RandomErasing(p=0.25),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size / 0.875), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_classification_loaders(data_path, image_size=224, batch_size=128, workers=8):
    root = Path(data_path)
    train_set = datasets.ImageFolder(root / "train", transform=build_imagenet_transforms(image_size, True))
    val_set = datasets.ImageFolder(root / "val", transform=build_imagenet_transforms(image_size, False))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader


class ADE20K(torch.utils.data.Dataset):
    def __init__(self, root, split="training", image_size=512):
        from PIL import Image

        self.Image = Image
        self.root = Path(root)
        self.image_size = image_size
        image_dir = self.root / "images" / split
        mask_dir = self.root / "annotations" / split
        self.images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        self.masks = [mask_dir / f"{path.stem}.png" for path in self.images]
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.mask_transform = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.Image.open(self.images[index]).convert("RGB")
        mask = self.Image.open(self.masks[index])
        image = self.image_transform(image)
        mask = torch.as_tensor(list(self.mask_transform(mask).getdata()), dtype=torch.long)
        mask = mask.view(self.image_size, self.image_size) - 1
        mask = torch.where(mask < 0, torch.full_like(mask, 255), mask)
        return image, mask


def build_segmentation_loaders(data_path, image_size=512, batch_size=16, workers=8):
    train_set = ADE20K(data_path, "training", image_size)
    val_set = ADE20K(data_path, "validation", image_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader
