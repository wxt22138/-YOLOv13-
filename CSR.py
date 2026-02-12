
import torch
import torch.nn as nn
import torch.nn.functional as F

class CSREnhanced(nn.Module):
    """
    完全保留你原有逻辑，仅确保branch属性存在
    """
    def __init__(self, channels, e_lambda_base=1e-5, reduction=4, branch='cls'):
        super(CSREnhanced, self).__init__()
        self.e_lambda_base = e_lambda_base
        self.channels = channels
        self.branch = branch  # 保留原有定义
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        self.activation = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.1))  

    def forward(self, x, is_cls_branch=False):
        # 完全保留原有forward逻辑
        if (self.branch == 'cls' and not is_cls_branch) or (self.branch == 'reg' and is_cls_branch):
            return x
        
        b, c, h, w = x.size()
        
        x_var = x.var(dim=[2, 3], keepdim=True)
        dynamic_lambda = self.e_lambda_base * (1 + torch.exp(-x_var))
        
        n = w * h - 1
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - x_mean).pow(2)
        denominator = 8 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + dynamic_lambda)
        energy = x_minus_mu_square / denominator + 0.5
        spatial_att = self.activation(energy)
        
        y = self.avg_pool(x).view(b, c)
        channel_att = self.fc(y).view(b, c, 1, 1)
        
        joint_att = spatial_att * channel_att * 0.5
        x_out = x + self.alpha * (x * joint_att)
        
        return x_out

# 兼容层：仅添加branch属性兜底，无任何逻辑改动
class CSR(CSREnhanced):
    def __init__(self, e_lambda=1e-5, branch='cls'):
        super().__init__(channels=0, e_lambda_base=e_lambda, branch=branch)
        # 强制兜底：确保无论如何self.branch都存在
        self.branch = branch
    
    def forward(self, x, is_cls_branch=False):
        # 兜底：如果加载权重后branch属性丢失，强制赋值
        if not hasattr(self, 'branch'):
            self.branch = 'cls'
        
        # 完全保留原有动态初始化逻辑
        if self.channels == 0:
            self.channels = x.size(1)
            self.fc = nn.Sequential(
                nn.Linear(self.channels, self.channels // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(self.channels // 4, self.channels, bias=False),
                nn.Sigmoid()
            ).to(x.device)
        return super().forward(x, is_cls_branch)
