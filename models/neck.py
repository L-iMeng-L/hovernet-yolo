# models/neck.py
import torch
import torch.nn as nn
from models.backbone import CBL, CSPBlock   # 相对包路径

class PAN(nn.Module):
    """
    Path Aggregation Network
    in_chs = [256, 512, 1024]对应 backbone x2/x3/x4
    out_ch = 256
    输出三尺度：
      p2: /8,  128ch
      p3: /16, 256ch
      p4: /32, 256ch
    """
    def __init__(self, in_chs=(256, 512, 1024), out_ch=256):
        super().__init__()
        # ---------- top-down ----------
        self.lat4= CBL(in_chs[2], out_ch, 1, 1, 0)
        self.up1     = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse1   = CSPBlock(out_ch + in_chs[1], out_ch, n=1)   # 256+512=768→256

        self.lat3    = CBL(out_ch, out_ch // 2, 1, 1, 0)
        self.up2     = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse2   = CSPBlock(out_ch // 2 + in_chs[0], out_ch // 2, n=1)  # 128+256=384→128

        # ---------- bottom-up ----------
        self.down1   = CBL(out_ch // 2, out_ch // 2, 3, 2, 1)
        self.fuse3   = CSPBlock(out_ch // 2 + out_ch, out_ch, n=1)           # 128+256=384→256

        self.down2   = CBL(out_ch, out_ch, 3, 2, 1)
        self.fuse4   = CSPBlock(out_ch + in_chs[2], out_ch, n=1)             # 256+1024=1280→256

    def forward(self, feats):
        x2, x3, x4 = feats   # /8-256, /16-512, /32-1024

        # top-down
        p4 = self.lat4(x4)                # /32, 256
        p3 = self.fuse1(torch.cat([self.up1(p4), x3], 1))  # /16, 256
        p2 = self.fuse2(torch.cat([self.up2(self.lat3(p3)), x2], 1))  # /8, 128

        # bottom-up
        p3 = self.fuse3(torch.cat([self.down1(p2), p3], 1))   # /16, 256
        p4 = self.fuse4(torch.cat([self.down2(p3), x4], 1))   # /32, 256

        return [p2, p3, p4]   # 128ch, 256ch, 256ch