import torch
import torch.nn as nn

class SCE(nn.Module):
    """
    修复版Coordinate Attention
    - 恢复平均池化（更适合小目标）
    - 去除多余插值
    - 使用SiLU保持与YOLO一致
    """
    def __init__(self, c1, c2=None, reduction=16):
        super().__init__()
        c2 = c2 or c1
        mip = max(8, c1 // reduction)
        
        # 使用平均池化，保留小目标的全局信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(c1, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)  # 与YOLO保持一致
        
        self.conv_h = nn.Conv2d(mip, c2, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, c2, 1, bias=False)
        
    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        
        # 分别池化
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]
        
        # 拼接处理
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        # 分割
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力（直接广播，不需要插值）
        a_h = self.conv_h(x_h).sigmoid()  # [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [B, C, 1, W]
        
        # 广播相乘（自动扩展维度）
        return identity * a_h * a_w
