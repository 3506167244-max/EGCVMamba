# EGCVMamba 训练指南（4×RTX 5090）

本文档覆盖 ImageNet-1K 分类训练和 ADE20K 语义分割微调。仓库不会把论文草稿中的数字当成已复现结果；训练产生的 `metrics.jsonl`、`best.pth` 和验证集指标才是最终依据。

## 1. 推荐环境

建议使用 Ubuntu 22.04/24.04、4 张 32 GB RTX 5090、Python 3.11、PyTorch 2.12.1 和 CUDA 13.0 wheel。PyTorch 官方的 2.12 发布说明建议 Blackwell GPU 使用 CUDA 13.0+ wheel，并要求 Linux 驱动至少为 580.65.06；版本命令可在 [PyTorch 官方安装页](https://pytorch.org/get-started/previous-versions/) 和 [PyTorch 2.12 发布说明](https://pytorch.org/blog/pytorch-2-12-release-blog/) 核对。

先检查驱动和四张卡：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
```

创建环境：

```bash
conda create -n egcvmamba python=3.11 -y
conda activate egcvmamba

pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130

cd /path/to/EGCVMamba
pip install -r requirements.txt
pip install -e . --no-deps
```

PyTorch wheel 已携带 CUDA 运行时。本仓库的 SS2D 是纯 PyTorch 实现，不需要单独编译 `mamba-ssm`、Triton 或 selective-scan CUDA 扩展。

验证环境：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY

pytest -q
```

预期测试结果为 3 个测试全部通过。如果 `torch.cuda.device_count()` 不是 4，先解决驱动、容器 GPU 映射或 `CUDA_VISIBLE_DEVICES`，不要直接开始正式训练。

## 2. 数据目录

ImageNet-1K 使用 `torchvision.datasets.ImageFolder`，训练集和验证集都必须按类别建子目录：

```text
/data/imagenet/
  train/
    n01440764/
      *.JPEG
    ...
  val/
    n01440764/
      *.JPEG
    ...
```

官方原始验证集如果还是平铺文件，需要先按官方标签映射整理成类别子目录。

ADE20K 目录应为：

```text
/data/ADEChallengeData2016/
  images/
    training/
    validation/
  annotations/
    training/
    validation/
```

## 3. ImageNet-1K 四卡训练

优先训练 Small：它通常比 Tiny 更有精度上限，同时训练成本明显低于 Base/Large。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node=4 \
  tools/train_classification.py \
  --config configs/classification/egcvmamba_small.yaml \
  --data-path /data/imagenet \
  --output outputs/imagenet/egcvmamba_small
```

也可使用封装脚本：

```bash
bash scripts/train_imagenet.sh \
  configs/classification/egcvmamba_small.yaml \
  /data/imagenet \
  outputs/imagenet/egcvmamba_small
```

默认配置按“每张 GPU 的 batch”解释：

| 模型 | 每卡 batch | 4 卡全局 batch | 自动缩放后的初始 LR | DropPath |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 512 | 2048 | 2e-3 | 0.05 |
| Small | 256 | 1024 | 1e-3 | 0.10 |
| Base | 192 | 768 | 7.5e-4 | 0.20 |
| Large | 128 | 512 | 5e-4 | 0.30 |

这些是 32 GB 5090 的起始值，不是显存承诺；CPU 解码速度、驱动版本和模型变体都会影响可用 batch。若 OOM，把对应 YAML 的 `train.batch_size` 改为更小的偶数。脚本会按

```text
实际学习率 = YAML 中的 lr × 全局 batch / reference_batch_size
```

自动缩放学习率，因此通常只需改 batch，不要同时手工改 LR。

分类默认训练 300 epoch，使用 20 epoch warm-up、AdamW、BF16、channels-last、RandAugment、Mixup、CutMix、Random Erasing、label smoothing 和 EMA。最佳模型按 EMA 的验证 Top-1 保存。

## 4. 断点续训和评估

同一个 `--output` 下存在 `last.pth` 时，默认会自动续训：

```bash
torchrun --standalone --nproc_per_node=4 \
  tools/train_classification.py \
  --config configs/classification/egcvmamba_small.yaml \
  --data-path /data/imagenet \
  --output outputs/imagenet/egcvmamba_small
```

也可以显式指定：

```bash
torchrun --standalone --nproc_per_node=4 \
  tools/train_classification.py \
  --config configs/classification/egcvmamba_small.yaml \
  --data-path /data/imagenet \
  --resume outputs/imagenet/egcvmamba_small/last.pth
```

单卡评估默认读取 EMA 权重：

```bash
python tools/evaluate_classification.py \
  --config configs/classification/egcvmamba_small.yaml \
  --checkpoint outputs/imagenet/egcvmamba_small/best.pth \
  --data-path /data/imagenet \
  --weights ema
```

输出目录包含：

```text
best.pth       # 按 EMA Top-1 选择的最佳 checkpoint
last.pth       # 最近一个 epoch，可完整续训
metrics.json   # 最近一个 epoch 的摘要
metrics.jsonl  # 全部 epoch 的逐行历史，建议用于画图和论文表格
```

Mixup/CutMix 开启时，训练阶段的 Top-1 只按原始硬标签粗略统计，不适合作为最终判断；重点观察 `ema_val.acc1`、`ema_val.acc5` 和验证损失。

## 5. ADE20K 四卡微调

为了获得合理 mIoU，先完成同变体的 ImageNet 分类训练，再加载其 EMA backbone。不要把 ADE20K 从随机初始化训练的结果和预训练微调结果混在一起比较。

Small 示例：

```bash
torchrun --standalone --nproc_per_node=4 \
  tools/train_segmentation.py \
  --config configs/segmentation/ade20k_small.yaml \
  --data-path /data/ADEChallengeData2016 \
  --pretrained outputs/imagenet/egcvmamba_small/best.pth \
  --output outputs/ade20k/egcvmamba_small
```

或使用脚本：

```bash
bash scripts/train_ade20k.sh \
  configs/segmentation/ade20k_small.yaml \
  /data/ADEChallengeData2016 \
  outputs/imagenet/egcvmamba_small/best.pth \
  outputs/ade20k/egcvmamba_small
```

分割默认使用随机缩放、512×512 成对裁剪、水平翻转和颜色扰动；backbone LR 是 decoder LR 的 0.1 倍。Tiny 每卡 batch 为 16，Small 每卡 batch 为 8。分割同样支持相同输出目录自动续训。

四卡验证最佳 checkpoint：

```bash
torchrun --standalone --nproc_per_node=4 \
  tools/train_segmentation.py \
  --config configs/segmentation/ade20k_small.yaml \
  --data-path /data/ADEChallengeData2016 \
  --resume outputs/ade20k/egcvmamba_small/best.pth \
  --eval-only
```

## 6. 常见问题

### CUDA 架构不支持或 `no kernel image`

确认驱动满足 CUDA 13 wheel 要求，并确认安装的是 `cu130` 而非过旧 wheel：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### 训练一开始 OOM

先减小每卡 batch；保持 `reference_batch_size` 不变，让代码自动调整 LR。仍然 OOM 时再把 `augmentation.prefetch_factor` 从 4 改为 2（它主要影响主机内存），或暂时关闭 `channels_last` 排查算子问题。

### GPU 利用率周期性掉到 0

通常是图片解码或磁盘吞吐不足。把数据放到本地 NVMe，逐步调高 `train.workers`，并确认主机内存足够。`workers` 是每个训练进程的数量，四卡总 worker 数等于配置值乘 4。

### DDP 卡住

先用 `nvidia-smi` 确认四卡都可见，再设置：

```bash
export NCCL_DEBUG=INFO
```

查看最先报错的 rank。PyTorch 官方推荐单机多卡使用一个进程对应一张 GPU 的 DDP/NCCL；本仓库的 `torchrun` 入口按这一方式实现。

### Loss 为 NaN

先把 YAML 中 `precision: bf16` 临时改成 `fp32` 做定位。如果 FP32 正常，再检查驱动/PyTorch 版本；若两者都 NaN，把基础 LR 降低 20% 到 50%，并检查数据中是否有损坏图片或非法标签。

## 7. 关于最终指标

代码和超参数只能提供可靠的训练起点，不能在未实际跑完数据集的情况下保证某个 Top-1 或 mIoU。建议至少保留三次不同随机种子的完整日志，报告均值、方差、参数量和实际计算量，并清楚区分单次最好结果与平均结果。仓库当前的纯 PyTorch SS2D 优先保证可训练性和新显卡兼容性；`fvcore` 对循环状态更新中的部分逐元素算子不能完整计数，因此 profiler 给出的 FLOPs 只能作近似参考。
