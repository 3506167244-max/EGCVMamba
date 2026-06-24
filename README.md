# EGCVMamba

**EGCVMamba: Efficient Gated Convolution-Mamba Hybrid Architecture for Visual Recognition**

EGCVMamba is a lightweight visual backbone for resource-limited recognition tasks. It combines reparameterized convolutional stems, gated CNN blocks, hierarchical multi-scale fusion, and an efficient 2D selective-scan style block in a stage-adaptive architecture.

**Authors:** HITSZ Mingshuai Chen, HKUST(GZ) Yue Jia, Rice University Enhao Peng

## Highlights

- Stage-adaptive CNN-Mamba hybrid design for edge visual recognition.
- Reparameterized stem and downsampling blocks for efficient deployment.
- AlphaBlock for fine-grained local feature extraction in high-resolution stages.
- BetaBlock with hierarchical gated fusion and channel aggregation for medium-resolution features.
- GammaBlock with recursive gated convolution and local kernel fusion for multi-scale local perception.
- EVSSBlock with efficient 2D selective-scan style global context modeling in the final low-resolution stage.
- Unified classification and semantic segmentation code path.

## Architecture

| Stage | Resolution | Module | Role |
| --- | --- | --- | --- |
| Stem | 1/4 | Reparameterized Stem | Initial feature extraction |
| Stage 1 | 1/4 | AlphaBlock | Local texture learning |
| Stage 2 | 1/8 | AlphaBlock + BetaBlock | Intermediate semantic refinement |
| Stage 3 | 1/16 | BetaBlock + GammaBlock | Multi-scale local enhancement |
| Stage 4 | 1/32 | EVSSBlock | Global context modeling |

## Installation

```bash
git clone https://github.com/3506167244-max/EGCVMamba.git
cd EGCVMamba
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
print(y.shape)  # torch.Size([1, 1000])
```

## Sanity Check

Before training, run the following checks to make sure the repository and environment are correctly installed:

```bash
python -m py_compile egcvmamba/models/egcvmamba.py
python -m py_compile egcvmamba/models/blocks.py
python -m py_compile egcvmamba/models/layers.py
python -m py_compile egcvmamba/models/segmentation.py
python -m py_compile egcvmamba/data.py
python -m py_compile egcvmamba/engine.py
pytest
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
python tools/train_classification.py \
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
python tools/train_segmentation.py \
  --config configs/segmentation/ade20k_tiny.yaml \
  --data-path /path/to/ADEChallengeData2016
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

## Reproducibility Notes

The released code is intended to make the model structure, training pipeline, and evaluation entry points clear and easy to inspect. Full pretrained checkpoints and detailed training logs will be released after acceptance. The reported numbers in the paper should be reproduced with the corresponding configuration files, dataset versions, image resolution, random seed, and training schedule.

## Citation

```bibtex
@inproceedings{chen2026egcvmamba,
  title={EGCVMamba: Efficient Gated Convolution-Mamba Hybrid Architecture for Visual Recognition},
  author={Chen, Mingshuai and Jia, Yue and Peng, Enhao},
  booktitle={ICONIP},
  year={2026}
}
```
