import torch
import torch.nn as nn

# -------------------------- 1. Addition --------------------------
class AdditionFusion(nn.Module):
    def forward(self, x, y):
        return x + y  # 无参数

# -------------------------- 2. Element-wise Multiply --------------------------
class ElementwiseMultiplyFusion(nn.Module):
    def forward(self, x, y):
        return x * y  # 无参数

# -------------------------- 3. Cross-Attention (Bias=False 以匹配参数估计) --------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            bias=False  # 关闭偏置以匹配参数估计
        )

    def forward(self, x, y):
        attn_output, _ = self.cross_attn(x, y, y)
        return attn_output

# -------------------------- 4. Gated Fusion --------------------------
class GatedFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),  # 参数: (1024*512 + 512) = 524,800
            nn.Sigmoid()
        )

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=-1)
        gate = self.gate(concat)
        return gate * x + (1 - gate) * y

# -------------------------- 5. Concatenation + Linear --------------------------
class ConcatLinearFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)  # 参数: (1024*512 + 512) = 524,800

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=-1)
        return self.proj(concat)

# -------------------------- 参数计算与测试 --------------------------
if __name__ == "__main__":
    # 初始化模型
    fusion_methods = {
        "Addition": AdditionFusion(),
        "ElementwiseMultiply": ElementwiseMultiplyFusion(),
        "CrossAttention": CrossAttentionFusion(),
        "GatedFusion": GatedFusion(),
        "ConcatLinear": ConcatLinearFusion()
    }

    # 测试输入
    x = torch.randn(64, 200, 512)
    y = torch.randn(64, 200, 512)

    # 遍历所有方法并计算参数量
    print("=" * 60)
    print(f"{'Fusion Method':<25} | {'Output Shape':<15} | {'Params Added':<12}")
    print("=" * 60)
    for name, model in fusion_methods.items():
        z = model(x, y)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:<25} | {str(z.shape):<15} | {params:<12,}")
    print("=" * 60)