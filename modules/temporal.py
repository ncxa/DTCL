import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class Temporal(Module):
    def __init__(self, input_size, out_size, num_layers=3, dropout=0., activation=nn.ReLU()):
        super(Temporal, self).__init__()

        self.num_layers = num_layers
        self.in_dim = input_size  # 输入特征维度
        self.out_dim = out_size  # 输出特征维度
        self.dropout = dropout
        self.activation = activation

        # 卷积层序列 (卷积层 + 批量归一化 + 激活函数)
        # 每一层卷积的输入通道数等于上一层的输出通道数
        self.tconvs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_size if i == 0 else out_size  # 第一层的输入通道数是 input_size，后续层输入通道数是 out_size
            self.tconvs.append(
                nn.Conv1d(in_channels=in_channels, out_channels=out_size, kernel_size=3, padding=1)
            )

        # 批量归一化
        # self.bns = nn.ModuleList([nn.BatchNorm1d(out_size) for _ in range(num_layers)])

    def forward(self, x):
        # 假设输入的形状是 (batch_size, seq_len, d_feature)，需要调整为 (batch_size, d_feature, seq_len) 来适应 Conv1d
        x = x.permute(0, 2, 1)  # 转换成 (batch_size, d_feature, seq_len)

        # 对每一层卷积进行操作
        for tconv in self.tconvs:
            x = self.activation(tconv(x))

        # 将输出转回 (batch_size, seq_len, out_size) 的形状
        x = x.permute(0, 2, 1)  # 转换回 (batch_size, seq_len, out_size)

        return x  # 返回特征图


def main():
    x = torch.randn(5, 100, 1024).cuda()  # 输入张量 (batch_size, seq_len, d_feature)

    # 实例化 Temporal 模型
    embedding = Temporal(1024, 512).cuda()  # 输入 1024 特征，输出 512 特征

    out = embedding(x)  # 前向传播

    print(out.shape)  # 输出形状，应该是 (5, 100, 512)


if __name__ == '__main__':
    main()
