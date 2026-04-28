# models/yolo11_backbone.py
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k//2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class YOLO11nBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem
        self.stem = Conv(3, 64, 3, 2)  # /2
        
        # Stage 1
        self.stage1 = nn.Sequential(
            Conv(64, 128, 3, 2),  # /4
            C3k2(128, 128, 1, True),
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            Conv(128, 256, 3, 2),  # /8
            C3k2(256, 256, 2, True),
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            Conv(256, 512, 3, 2),  # /16
            C3k2(512, 512, 2, True),
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            Conv(512, 512, 3, 2),  # /32
            C3k2(512, 512, 1, True),
            SPPF(512, 512, 5),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # /8, 256ch
        p4 = self.stage3(p3)  # /16, 512ch
        p5 = self.stage4(p4)  # /32, 512ch
        return [p3, p4, p5]