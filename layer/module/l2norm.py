import numpy as np
import torch

from lib import *


# Norm: Chuẩn hóa để tính ra một đại lượng biểu diễn độ lớn của một vector
# L2Norm: Euclid distance
class L2Norm(nn.Module):
    def __init__(self, in_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels))
        self.scale = scale
        self.reset_parameter()
        self.eps = 1e-10

    def reset_parameter(self):
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        # x.size() = (batch_size, channel, height, weight)
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # norm trên từng layer
        x = torch.div(x, norm)
        # weight.size() = (512) -> (1,512,1,1)
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        # norm trên toàn bộ layer
        return weights * x
