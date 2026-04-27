# models/backbone.py
# 只需在 CSPDarknet.forward() 末尾 return三层特征
# 如果原来只return [x4]，改成 return [x2, x3, x4]
# 下面是完整示例（假设原来的 CBL / CSPBlock 保持不动）

import torch
import torch.nn as nn

class CBL(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),)
    def forward(self, x): return self.conv(x)

class ResUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.cv = nn.Sequential(CBL(ch, ch, 3, 1, 1), CBL(ch, ch, 3, 1, 1))
    def forward(self, x): return x + self.cv(x)

class CSPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n=1):
        super().__init__()
        mid = out_ch // 2
        self.cv1  = CBL(in_ch, mid, 1, 1, 0)
        self.cv2  = CBL(in_ch, mid, 1, 1, 0)
        self.res  = nn.Sequential(*[ResUnit(mid) for _ in range(n)])
        self.cat= CBL(mid * 2, out_ch,1, 1, 0)
    def forward(self, x):
        return self.cat(torch.cat([self.res(self.cv1(x)), self.cv2(x)], 1))

class CSPDarknet(nn.Module):
    """
    输出[x2, x3, x4]：x2: /8,256ch
      x3: /16, 512ch
      x4: /32, 1024ch
    base_ch=64时与原来一致
    """
    def __init__(self, base_ch=64, base_depth=3):
        super().__init__()
        b = base_ch
        # stem: /2
        self.stem  = CBL(3, b, 6, 2, 2)
        # stage1: /4
        self.stage1 = nn.Sequential(CBL(b, b*2, 3, 2, 1),CSPBlock(b*2, b*2, base_depth))
        # stage2: /8  → x2
        self.stage2 = nn.Sequential(CBL(b*2, b*4, 3, 2, 1),
                                     CSPBlock(b*4, b*4, base_depth*3))
        # stage3: /16 → x3
        self.stage3 = nn.Sequential(CBL(b*4, b*8, 3, 2, 1),
                                     CSPBlock(b*8, b*8, base_depth*3))
        # stage4: /32 → x4
        self.stage4 = nn.Sequential(CBL(b*8, b*16, 3, 2, 1),
                                     CSPBlock(b*16, b*16, base_depth))

    def forward(self, x):
        x= self.stem(x)
        x  = self.stage1(x)
        x2 = self.stage2(x)   # /8,  256
        x3 = self.stage3(x2)  # /16, 512
        x4 = self.stage4(x3)  # /32, 1024
        return [x2, x3, x4]   # ← 三层全部返回，供Decoder 跳跃连接