import torch
from models.seg_model import HoverSegModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HoverSegModel(base_ch=64, num_classes=5).to(device)
model.eval()  # 推理模式

model.load_state_dict(torch.load("model_weights_only.pth", map_location="cpu"))


total_params = sum(p.numel() for p in model.parameters())  # 总参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数

print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"理论模型大小（FP32）: {total_params * 4 / 1024 / 1024:.2f} MB")
# 说明：FP32每个float占4字节，所以乘以4，再除以1024转MB