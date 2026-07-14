# EGCVMamba

Official PyTorch implementation for **EGCVMamba: Efficient Gated Convolution-Mamba Hybrid Architecture for Visual Recognition**.

Mingshuai Chen<sup>1</sup>, Yue Jia<sup>2</sup>, Enhao Peng<sup>3</sup>

<sup>1</sup>Harbin Institute of Technology, Shenzhen  
<sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)  
<sup>3</sup>Rice University

EGCVMamba is a lightweight visual backbone for resource-limited recognition tasks. It combines reparameterized convolutional stems, gated CNN blocks, hierarchical multi-scale fusion, and an efficient 2D selective scan block in a stage-adaptive architecture.

## Highlights

- Stage-adaptive CNN-Mamba hybrid design for edge visual recognition.
- Reparameterized stem and downsampling blocks for efficient deployment.
- AlphaBlock for fine-grained local feature extraction in high-resolution stages.
- BetaBlock with hierarchical gated fusion and channel aggregation for medium-resolution features.
- GammaBlock with recursive gated convolution and local kernel fusion for multi-scale local perception.
- EVSSBlock with 2D selective scan for long-range global modeling in the final low-resolution stage.
- Unified classification and semantic segmentation code path.

## Architecture

| Stage | Resolution | Module | Role |
| --- | --- | --- | --- |
| Stem | 1/4 | Reparameterized Stem | Initial feature extraction |
| Stage 1 | 1/4 | AlphaBlock | Local texture learning |
| Stage 2 | 1/8 | AlphaBlock + BetaBlock | Intermediate semantic refinement |
| Stage 3 | 1/16 | BetaBlock + GammaBlock | Multi-scale local enhancement |
| Stage 4 | 1/32 | EVSSBlock | Global context modeling |

## Model Configurations

| Model | Channels | Blocks |
| --- | --- | --- |
| EGCVMamba-T | [64, 96, 192, 384] | [1, 1, 1, 1] |
| EGCVMamba-S | [64, 128, 256, 512] | [2, 3, 4, 2] |
| EGCVMamba-B | [80, 160, 320, 640] | [3, 4, 6, 3] |
| EGCVMamba-L | [96, 192, 384, 768] | [4, 6, 8, 4] |

The paper-draft metrics have not been reproduced and are therefore not presented as verified results here. Use the generated validation logs and checkpoints as the source of truth for each run.

## Pretrained Models

Pretrained checkpoints and complete training logs will be released after paper acceptance.

| Model | ImageNet-1K | ADE20K | Status |
| --- | --- | --- | --- |
| EGCVMamba-T | Pending | Pending | Release after acceptance |
| EGCVMamba-S | Pending | Pending | Release after acceptance |
| EGCVMamba-B | Pending | - | Release after acceptance |
| EGCVMamba-L | Pending | - | Release after acceptance |

The repository does not currently provide pretrained weights. Files produced by local training are stored under `outputs/` and are excluded from version control.

For the recommended 4-GPU RTX 5090 environment, tuned recipes, resume/evaluation commands, and troubleshooting, see [TRAINING_ZH.md](TRAINING_ZH.md).

## Installation

```bash
git clone https://github.com/3506167244-max/EGCVmamba.git
cd EGCVmamba
conda create -n egcvmamba python=3.10 -y
conda activate egcvmamba
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
import torch
from egcvmamba.models import build_model

model = build_model("tiny", num_classes=1000)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(y.shape)
```

## ImageNet-1K Training

Expected dataset layout:

```text
/path/to/imagenet
  train/
    class_name/
      image.jpg
  val/
    class_name/
      image.jpg
```

Training:

```bash
torchrun --standalone --nproc_per_node=4 tools/train_classification.py \
  --config configs/classification/egcvmamba_tiny.yaml \
  --data-path /path/to/imagenet
```

Evaluation:

```bash
python tools/evaluate_classification.py \
  --config configs/classification/egcvmamba_tiny.yaml \
  --checkpoint outputs/imagenet/egcvmamba_tiny/best.pth \
  --data-path /path/to/imagenet
```

## ADE20K Semantic Segmentation

Expected dataset layout:

```text
/path/to/ADEChallengeData2016
  images/
    training/
    validation/
  annotations/
    training/
    validation/
```

Training:

```bash
torchrun --standalone --nproc_per_node=4 tools/train_segmentation.py \
  --config configs/segmentation/ade20k_tiny.yaml \
  --data-path /path/to/ADEChallengeData2016 \
  --pretrained outputs/imagenet/egcvmamba_tiny/best.pth
```

## Profiling

```bash
python tools/profile_model.py --variant tiny --image-size 224
```

## ONNX Export

```bash
python tools/export_onnx.py \
  --variant tiny \
  --checkpoint outputs/imagenet/egcvmamba_tiny/best.pth \
  --output checkpoints/egcvmamba_tiny.onnx
```

## Repository Layout

```text
egcvmamba/
  models/
    blocks.py
    egcvmamba.py
    layers.py
    segmentation.py
  data.py
  engine.py
  utils.py
configs/
  classification/
  segmentation/
tools/
  train_classification.py
  evaluate_classification.py
  train_segmentation.py
  profile_model.py
  export_onnx.py
tests/
  test_model_shapes.py
```

## Citation

```bibtex
@inproceedings{chen2026egcvmamba,
  title={EGCVMamba: Efficient Gated Convolution-Mamba Hybrid Architecture for Visual Recognition},
  author={Chen, Mingshuai and Jia, Yue and Peng, Enhao},
  booktitle={ICONIP},
  year={2026}
}
```

## License

This project is released under the MIT license.
