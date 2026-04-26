import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from model import EGCVMamba_tiny,EGCVMamba_base,EGCVMamba_small,EGCVMamba_large
import signal
import sys
import traceback


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

TERMINATE = False
try:
    import wandb
except ImportError:
    wandb = None


def signal_handler(sig, frame):
    global TERMINATE
    print("\n⚠️  收到终止信号，准备优雅退出...")
    TERMINATE = True


signal.signal(signal.SIGINT, signal_handler)


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        # EMA参数移到GPU（避免CPU/GPU数据拷贝，同时节省CPU内存）
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    new_val = (self.decay * self.shadow[name].data
                               + (1 - self.decay) * param.detach().clamp(-10, 10))
                    self.shadow[name].data = new_val

    def apply_shadow(self):
        self.backup = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # 默认单卡

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    return {
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "is_main_process": (rank == 0)
    }


def check_nan(tensor, name="tensor"):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        rank = os.environ.get("RANK", "unknown")
        print(f"\n🚨 进程{rank} 检测到 {name} 包含NaN/Inf! 已自动裁剪")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)
        return True, tensor
    return False, tensor


def clean_gradient(model):

    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan, cleaned_grad = check_nan(param.grad, f"Grad_{name}")
            if has_nan:
                param.grad.data = cleaned_grad
                param.grad.data = param.grad.data.clamp(-5, 5)



def apply_gradient_checkpointing(model):

    if hasattr(model, 'apply_gradient_checkpointing'):
        model.apply_gradient_checkpointing()
    else:
        # 通用梯度检查点适配
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
    return model

def validate(model, val_loader, criterion, device, epoch, proc_info, args, ema=None):
    if TERMINATE:
        return 0, 0, 0

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False, disable=not proc_info["is_main_process"])
        for images, labels in pbar:
            if TERMINATE:
                pbar.close()
                return 0, 0, 0

            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # 验证阶段使用float16，进一步降低显存
            with autocast(dtype=torch.float16):
                outputs = model(images)
                has_nan, outputs = check_nan(outputs, f"Val_Epoch{epoch}_outputs")
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if proc_info["is_main_process"]:
                pbar.set_postfix({'val_loss': loss.item(), 'val_acc': correct / total})

        # EMA验证
        ema_acc = 0.0
        if proc_info["is_main_process"] and ema is not None:
            ema.apply_shadow()
            pbar_ema = tqdm(val_loader, desc=f"Epoch {epoch} EMA Val", leave=False)
            ema_correct, ema_total = 0, 0
            for images, labels in pbar_ema:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(dtype=torch.float16):
                    outputs = model(images)
                _, predicted = outputs.max(1)
                ema_total += labels.size(0)
                ema_correct += predicted.eq(labels).sum().item()
                pbar_ema.set_postfix({'ema_acc': ema_correct / ema_total})
            ema_acc = 100. * ema_correct / ema_total if ema_total > 0 else 0
            ema.restore()

    avg_val_loss = val_loss / total if total > 0 else 0
    val_acc = 100. * correct / total if total > 0 else 0

    if proc_info["is_main_process"]:
        print(f"Validation - Epoch: {epoch}, Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%, EMA Acc: {ema_acc:.2f}%")
        if args.use_wandb and wandb is not None:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
                "val/ema_accuracy": ema_acc,
                "epoch": epoch
            })

    return avg_val_loss, val_acc, ema_acc


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, proc_info, args,
                    ema=None):
    if TERMINATE:
        return 0, 0

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train (GPU{proc_info['local_rank']})", leave=False,
                disable=not proc_info["is_main_process"])
    for step, (images, labels) in enumerate(pbar):
        if TERMINATE:
            pbar.close()
            return 0, 0

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # 训练阶段使用float16，降低显存占用
        with autocast(dtype=torch.float16):
            outputs = model(images)
            has_nan, outputs = check_nan(outputs, f"Epoch{epoch}_Step{step}_outputs")
            if has_nan:
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = criterion(outputs, labels)
            loss = loss / args.accumulate_steps
            has_nan, loss = check_nan(loss, f"Epoch{epoch}_Step{step}_loss")
            if has_nan:
                optimizer.zero_grad(set_to_none=True)
                continue


        scaler.scale(loss).backward()

        if (step + 1) % args.accumulate_steps == 0:
            scaler.unscale_(optimizer)
            clean_gradient(model)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.clip_grad_norm,
                error_if_nonfinite=False
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if proc_info["is_main_process"] and ema is not None:
                ema.update()


        if proc_info["is_main_process"]:
            train_loss += loss.item() * images.size(0) * args.accumulate_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'train_loss': loss.item(), 'train_acc': correct / total})

    if scheduler is not None:
        scheduler.step()

    avg_train_loss = train_loss / total if total > 0 else 0
    train_acc = 100. * correct / total if total > 0 else 0

    if proc_info["is_main_process"]:
        print(
            f"Training - Epoch: {epoch}, Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")
        if args.use_wandb and wandb is not None:
            wandb.log({
                "train/loss": avg_train_loss,
                "train/accuracy": train_acc,
                "train/lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })

    return avg_train_loss, train_acc


def main_worker(args):
    global TERMINATE

    proc_info = init_distributed()
    local_rank = proc_info["local_rank"]
    rank = proc_info["rank"]
    is_main_process = proc_info["is_main_process"]
    device = torch.device(f'cuda:{local_rank}')

    if is_main_process:
        torch.manual_seed(args.seed)
        print(f"进程{rank}绑定GPU {local_rank} (主进程)")
        print(f"单卡batch_size: {args.batch_size}, 总卡数: {proc_info['world_size']}")
        print(f"总有效batch_size: {args.batch_size * proc_info['world_size'] * args.accumulate_steps}")
        print(f"LR={args.lr}, 梯度裁剪={args.clip_grad_norm}")


    from data import get_dataloaders
    train_loader_original, val_loader_original = get_dataloaders()
    train_dataset = train_loader_original.dataset
    val_dataset = val_loader_original.dataset

    train_sampler = None
    if proc_info["world_size"] > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=proc_info["world_size"],
            rank=rank,
            shuffle=True,
            drop_last=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=2
    )

    val_loader = None
    if is_main_process:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers // 2,
            pin_memory=True,
            persistent_workers=False
        )


    model = ReGeForceNet_tiny(num_classes=args.num_classes, drop_path_rate=0.05).to(device)

    model = apply_gradient_checkpointing(model)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.clamp(-1, 1)


    if proc_info["world_size"] > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    ema = None
    if is_main_process:
        ema = EMA(model.module if proc_info["world_size"] > 1 else model, decay=args.ema_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)

    scaled_lr = args.lr * args.batch_size * proc_info["world_size"] * args.accumulate_steps / 512
    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-6,
        foreach=False
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 1000
    )

    scaler = torch.cuda.amp.GradScaler(
        init_scale=2. ** 16,
        growth_interval=2000,
        backoff_factor=0.5
    )


    if is_main_process and args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args, save_code=True)


    best_val_acc = 0.0
    best_ema_acc = 0.0
    if is_main_process:
        print(f"\n🚀 开始32GB显存优化训练，共{args.epochs}轮...")

    for epoch in range(args.epochs):
        if TERMINATE:
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, proc_info, args, ema
        )

        if proc_info["is_main_process"] and (train_loss != train_loss or train_loss > 1e6):
            continue

        if is_main_process and val_loader is not None:
            val_model = model.module if proc_info["world_size"] > 1 else model
            val_loss, val_acc, ema_acc = validate(
                val_model, val_loader, criterion, device, epoch, proc_info, args, ema
            )
            if val_acc > best_val_acc and not TERMINATE:
                best_val_acc = val_acc
                save_path = os.path.join(args.save_dir, "best_regeforce_tiny.pth")
                torch.save(val_model.state_dict(), save_path)
                print(f" 最佳原模型保存: {save_path} (Val Acc: {best_val_acc:.2f}%)")

            if ema_acc > best_ema_acc and not TERMINATE:
                best_ema_acc = ema_acc
                save_path = os.path.join(args.save_dir, "best_regeforce_tiny_ema.pth")
                torch.save({
                    "model_state_dict": val_model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "epoch": epoch,
                    "best_ema_acc": best_ema_acc
                }, save_path)
                print(f" 最佳EMA模型保存: {save_path} (EMA Acc: {best_ema_acc:.2f}%)")

        torch.cuda.empty_cache()

    if is_main_process:
        if args.use_wandb and wandb is not None:
            wandb.run.summary["best_val_acc"] = best_val_acc
            wandb.run.summary["best_ema_acc"] = best_ema_acc
            wandb.finish()
        print(f" 最佳原模型精度: {best_val_acc:.2f}%")
        print(f"最佳EMA模型精度: {best_ema_acc:.2f}%")

    if proc_info["world_size"] > 1 and dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='32GB显存训练ReGeForceNet-Tiny (无OOM)')

    parser.add_argument('--batch_size', type=int, default=64,
                        )
    parser.add_argument('--num_workers', type=int, default=4,
                        )
    parser.add_argument('--accumulate_steps', type=int, default=2,
                       )
    parser.add_argument('--clip_grad_norm', type=float, default=5.0,
                      )


    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-3,
                        )
    parser.add_argument('--weight_decay', type=float, default=0.03,
                       )
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_32gb')


    parser.add_argument('--no_wandb', action='store_true', )
    parser.add_argument('--wandb_project', type=str, default='EGCVMamba')
    parser.add_argument('--wandb_run_name', type=str, default='EGCVMamba')

    args = parser.parse_args()
    args.use_wandb = not args.no_wandb


    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(args.save_dir, exist_ok=True)


    main_worker(args)


if __name__ == "__main__":
    main()