import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def build_imagenet_transforms(
    image_size=224,
    is_train=True,
    color_jitter=0.4,
    randaugment_ops=2,
    randaugment_magnitude=9,
    random_erasing=0.25,
):
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_magnitude),
                transforms.ColorJitter(color_jitter, color_jitter, color_jitter),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.RandomErasing(p=random_erasing),
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


def build_classification_loaders(
    data_path,
    image_size=224,
    batch_size=128,
    workers=8,
    distributed=False,
    rank=0,
    world_size=1,
    augmentation=None,
):
    root = Path(data_path)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet root must contain train/ and val/ directories, got: {root.resolve()}"
        )
    augmentation = augmentation or {}
    train_set = datasets.ImageFolder(
        train_dir,
        transform=build_imagenet_transforms(
            image_size,
            True,
            color_jitter=augmentation.get("color_jitter", 0.4),
            randaugment_ops=augmentation.get("randaugment_ops", 2),
            randaugment_magnitude=augmentation.get("randaugment_magnitude", 9),
            random_erasing=augmentation.get("random_erasing", 0.25),
        ),
    )
    val_set = datasets.ImageFolder(root / "val", transform=build_imagenet_transforms(image_size, False))
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
    loader_options = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = augmentation.get("prefetch_factor", 4)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        **loader_options,
    )
    return train_loader, val_loader


class ADE20K(torch.utils.data.Dataset):
    def __init__(self, root, split="training", image_size=512, scale_range=(0.5, 2.0)):
        from PIL import Image

        self.Image = Image
        self.root = Path(root)
        self.image_size = image_size
        self.training = split == "training"
        self.scale_range = scale_range
        image_dir = self.root / "images" / split
        mask_dir = self.root / "annotations" / split
        self.images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        self.masks = [mask_dir / f"{path.stem}.png" for path in self.images]
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        if not self.images:
            raise FileNotFoundError(f"No ADE20K images found in {image_dir.resolve()}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.Image.open(self.images[index]).convert("RGB")
        mask = self.Image.open(self.masks[index])
        if self.training:
            scale = random.uniform(*self.scale_range)
            target_short_side = max(int(self.image_size * scale), 1)
            resize_factor = target_short_side / min(image.height, image.width)
            height = max(int(image.height * resize_factor), 1)
            width = max(int(image.width * resize_factor), 1)
            image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [height, width], interpolation=InterpolationMode.NEAREST)
            pad_height = max(self.image_size - height, 0)
            pad_width = max(self.image_size - width, 0)
            if pad_height or pad_width:
                padding = [0, 0, pad_width, pad_height]
                image = TF.pad(image, padding, fill=(124, 116, 104))
                mask = TF.pad(mask, padding, fill=0)
            top, left, crop_height, crop_width = transforms.RandomCrop.get_params(
                image, output_size=(self.image_size, self.image_size)
            )
            image = TF.crop(image, top, left, crop_height, crop_width)
            mask = TF.crop(mask, top, left, crop_height, crop_width)
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            image = self.color_jitter(image)
        else:
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask = TF.pil_to_tensor(mask).squeeze(0).long() - 1
        mask = torch.where(mask < 0, torch.full_like(mask, 255), mask)
        return image, mask


def build_segmentation_loaders(
    data_path,
    image_size=512,
    batch_size=16,
    workers=8,
    distributed=False,
    rank=0,
    world_size=1,
    scale_range=(0.5, 2.0),
):
    train_set = ADE20K(data_path, "training", image_size, scale_range)
    val_set = ADE20K(data_path, "validation", image_size)
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
    options = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **options,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        **options,
    )
    return train_loader, val_loader
