"""Kimi 线性注意力模块使用示例。"""

import torch

from kimi_attention import KimiLinearAttention


def main() -> None:
    torch.manual_seed(2026)

    model = KimiLinearAttention(dim=128, num_heads=8, dropout=0.1)
    x = torch.randn(4, 32, 128)  # (batch, seq_len, dim)

    y = model(x)
    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
