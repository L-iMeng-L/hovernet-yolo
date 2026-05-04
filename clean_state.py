import torch
from collections import OrderedDict
from models.seg_model import HoverSegModel

checkpoint_path = "./hover_unet_weights.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu",weights_only=False)

model_state_dict = checkpoint["model_state"]

clean_state_dict = OrderedDict()
for k, v in model_state_dict.items():
    name = k.replace("module.", "")  # 移除DDP训练时自动添加的前缀
    clean_state_dict[name] = v

# 不包含优化器状态、epoch等信息
torch.save(clean_state_dict, "hover_unet_weights_only.pth")
print("权重文件已保存为 hover_unet_weights_only.pth")