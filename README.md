<div align="center">
<h3>EGCVMamba: Efficient Gated Convolution-Mamba Hybrid Architecture for Visual Recognition</h3>

Mingshuai Chen, Yue Jia, Enhao Peng  

Harbin Institute of Technology, Shenzhen, China <sup>1</sup>  
The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China <sup>2</sup>  
William Marsh Rice University (Rice University), Houston, USA <sup>3</sup>  

2025310439@stu.hit.edu.cn  

[Code](https://github.com/3506167244-max/SCLENet)
</div>

---

## Abstract
Resource-limited edge vision tasks require lightweight backbones with high accuracy and low complexity. Existing CNN-Mamba hybrid models (e.g., VCMamba) enhance local modeling capabilities but suffer from structural redundancy and excessive parameter overhead.

To address this challenge, we propose **EGCVMamba**, a stage-adaptive lightweight architecture tailored to the semantic richness and feature map resolution variations across downsampling stages. Our design unifies gated convolutions and optimized Mamba modules:
It employs a reparameterized stem and gated CNN blocks for fine-grained local learning in early high-resolution stages.
It leverages a lightweight EVSS-Block (2D selective scan Mamba) for efficient global modeling in the final low-resolution stage, eliminating the redundant stacking seen in prior works.

Extensive experiments on ImageNet-1K and ADE20K demonstrate that EGCVMamba outperforms VCMamba and other state-of-the-art lightweight models with lower computation:
- **EGCVMamba-T**: Achieves 72.1% Top-1 accuracy on ImageNet-1K with only 2.0M parameters.
- **EGCVMamba-S**: Attains 41.2% mIoU on ADE20K.

  <div align="center">
<h3>Prerequisites</h3>
python 3.9.25 
