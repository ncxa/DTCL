import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrupletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, reduction='mean'):
        super(QuadrupletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction

    def forward(self, anchor, positive, samples, negative):
        # 三元组1：anchor - positive - negative
        triplet1 = F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p,
                                         reduction=self.reduction)

        # 三元组2：anchor - samples - negative
        triplet2 = F.triplet_margin_loss(anchor, samples, negative, margin=self.margin, p=self.p,
                                         reduction=self.reduction)

        # 总损失
        return triplet1 + 0.01*triplet2

if __name__ == '__main__':
    # 创建四元组损失实例
    quadruplet_loss = QuadrupletMarginLoss(margin=1.0, p=2)

    # 创建随机的 anchor, positive, samples 和 negative 张量
    anchor = torch.randn(100, 128, requires_grad=True)
    positive = torch.randn(100, 128, requires_grad=True)
    samples = torch.randn(100, 128, requires_grad=True)
    negative = torch.randn(100, 128, requires_grad=True)

    # 计算四元组损失
    output = quadruplet_loss(anchor, positive, samples, negative)
    print(output)
