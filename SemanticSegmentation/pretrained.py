import torch
import os  # 新增：路径校验
from ImageClassification.model import EGCVMamba_tiny,EGCVMamba_small, EGCVMamba_base, EGCVMamba_large

weight_path = "/home/maxverstappen/projects/EGCVMamba/best_EGCVmamba_xxx.pth"
num_classes = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")


model = EGCVMamba_tiny(num_classes=num_classes,
                       drop_path_rate=0.05
                      ).to(device)

checkpoint = torch.load(
    weight_path,
    map_location=device,
    weights_only=True
)
model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()



def test_inference(model, device):

    dummy_input = torch.randn(1, 3, 224, 224).to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(dummy_input)
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float32):  # 改用float32
           output = model(dummy_input)
    else:
        output = model(dummy_input)
test_inference(model, device)