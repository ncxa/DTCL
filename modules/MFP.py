import torch
import torch.nn as nn
import torch.nn.functional as F

class mfp(nn.Module):
    def __init__(self, in_channels, num_layers=3, dropout_rate=0.):
        super(mfp, self).__init__()
        self.num_layers = num_layers
        # 定义多条路径的卷积、批量归一化、激活和 dropout 层
        self.paths = nn.ModuleList()
        kernel_sizes = [1, 5, 7]  # 定义多尺度卷积核尺寸

        for i in range(len(kernel_sizes)):
            layers = nn.ModuleList()
            for j in range(num_layers):
                kernel_size = kernel_sizes[i]
                padding = kernel_size // 2  # 确保卷积后长度不变
                if j == 0:
                    # 第一次卷积将通道数从 1024 降维到 512
                    layers.append(nn.Conv1d(in_channels, 512, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
                else:
                    # 后续卷积保持输入输出通道一致为 512
                    layers.append(nn.Conv1d(512, 512, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
                layers.append(nn.BatchNorm1d(512))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            self.paths.append(layers)

        # 可学习权重参数，用于加权每一条路径的最终特征
        self.path_weights = nn.Parameter(torch.ones(len(kernel_sizes)))  # 四条路径的可学习权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, z):
        x = x.permute(0, 2, 1)  # (batch_size, channels, length)
        z = z.permute(0, 2, 1)  # (batch_size, channels, length)
        y_fuse = 0
        path_weights = F.softmax(self.path_weights, dim=0)
        # 逐条路径处理
        for i, layers in enumerate(self.paths):
            y = x
            for layer in layers:
                y = layer(y)  # 按层执行卷积、归一化、激活和 dropout
            # 将每条路径的输出按路径的权重进行加权求和
            y_fuse += path_weights[i] * y

        # 通过 sigmoid 激活生成注意力权重
        y_fuse = self.sigmoid(y_fuse)
        # 应用注意力权重
        # out = z * y_fuse  # (batch_size, channels, length)
        out = z + z * y_fuse   # (batch_size, channels, length)
        out = out.permute(0, 2, 1)

        return out  # (batch_size, length, channels)


if __name__ == '__main__':
    x = torch.rand(64, 100, 1024)  # (batch_size, length, channels)
    z = torch.rand(64, 100, 512)  # (batch_size, length, channels)
    eca = mfp(1024, num_layers=3)
    result = eca(x,z)
    # print(eca)
    print("input.shape:", x.shape)
    print("output.shape:", result.shape)
