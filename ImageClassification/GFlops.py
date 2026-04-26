import torch
from thop import profile, clever_format
from model import EGCVMamba_tiny


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = torch.randn(1, 3, 224, 224)
print(f"最终推理设备: {device}")
print(f"输入尺寸: {input_tensor.shape}")


print("\n" + "="*60)
print("重参数化前统计 (训练态结构)")
print("="*60)


model_original = EGCVMamba_tiny(num_classes=100)
model_original.eval()


flops_original, params_original = profile(
    model_original,
    inputs=(input_tensor,),
    verbose=False
)

flops_original_g, params_original_m = clever_format([flops_original, params_original], "%.2f")
print(f"FLOPs:  {flops_original_g}")
print(f"Params: {params_original_m}")

print("\n" + "="*60)
print("正在执行重参数化...")
print("="*60)

model_reparam = model_original.reparameterize()
model_reparam.eval()
print("重参数化完成！")

print("\n" + "="*60)
print("重参数化后统计 (推理态结构)")
print("="*60)

flops_reparam, params_reparam = profile(
    model_reparam,
    inputs=(input_tensor,),
    verbose=False
)

# 格式化输出
flops_reparam_g, params_reparam_m = clever_format([flops_reparam, params_reparam], "%.2f")
print(f"FLOPs:  {flops_reparam_g}")
print(f"Params: {params_reparam_m}")

print("\n" + "="*60)
print("重参数化前后对比")
print("="*60)

flops_diff = (flops_original - flops_reparam) / flops_original * 100
params_diff = (params_original - params_reparam) / params_original * 100

print(f"{'指标':<10} | {'重参数化前':<12} | {'重参数化后':<12} | {'变化幅度':<10}")
print("-"*60)
print(f"{'FLOPs':<10} | {flops_original_g:<12} | {flops_reparam_g:<12} | ↓ {flops_diff:.1f}%")
print(f"{'Params':<10} | {params_original_m:<12} | {params_reparam_m:<12} | ↓ {params_diff:.1f}%")

print("\n" + "="*60)
print("额外：CUDA 推理速度测试")
print("="*60)

model_reparam_cuda = model_reparam.to(device)
input_tensor_cuda = input_tensor.to(device)

import time
warmup_iter = 10
test_iter = 100

with torch.no_grad():
    for _ in range(warmup_iter):
        _ = model_reparam_cuda(input_tensor_cuda)

torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    for _ in range(test_iter):
        _ = model_reparam_cuda(input_tensor_cuda)
torch.cuda.synchronize()
end_time = time.time()

fps = test_iter / (end_time - start_time)
print(f"测速完成！平均 FPS: {fps:.1f}")
