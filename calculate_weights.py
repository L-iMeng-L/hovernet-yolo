import torch
from models.seg_model import HoverSegModel

# 🔥 只初始化模型，不加载任何权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HoverSegModel(base_ch=64, num_classes=5).to(device)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"HoVer-UNet 总参数量: {total_params:,}")
print(f"HoVer-UNet 可训练参数量: {trainable_params:,}")
print(f"HoVer-UNet 模型大小（FP32）: {total_params * 4 / 1024 / 1024:.2f} MB")