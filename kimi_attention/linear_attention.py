from __future__ import annotations

import torch
from torch import Tensor, nn


class KimiLinearAttention(nn.Module):
    """Kimi-style multi-head linear attention module.

    This module replaces the quadratic softmax attention with a linearized
    formulation using a positive feature map:
        phi(x) = elu(x) + 1

    Attention is computed as:
        out_i = (phi(q_i) @ (phi(K)^T V)) / (phi(q_i) @ sum(phi(K)) + eps)

    Args:
        dim: Input embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Optional per-head dimension. Defaults to dim // num_heads.
        dropout: Dropout probability after output projection.
        bias: Whether to use bias in projection layers.
        eps: Numerical stability term for denominator.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")

        inner_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(dim, inner_dim, bias=bias)
        self.k_proj = nn.Linear(dim, inner_dim, bias=bias)
        self.v_proj = nn.Linear(dim, inner_dim, bias=bias)
        self.out_proj = nn.Linear(inner_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    @staticmethod
    def _feature_map(x: Tensor) -> Tensor:
        return torch.nn.functional.elu(x) + 1.0

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Query tensor of shape (batch, q_len, dim).
            context: Optional key/value tensor (batch, kv_len, dim). If None,
                self-attention is used with context=x.
            mask: Optional boolean or 0/1 tensor of shape (batch, kv_len)
                where 1/True means keep and 0/False means masked out.

        Returns:
            Tensor with shape (batch, q_len, dim).
        """
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B, L, D), got shape {tuple(x.shape)}")

        context = x if context is None else context
        if context.ndim != 3:
            raise ValueError(
                f"context must be 3D (B, L, D), got shape {tuple(context.shape)}"
            )

        bsz, q_len, _ = x.shape
        _, kv_len, _ = context.shape

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = self._feature_map(q)
        k = self._feature_map(k)

        if mask is not None:
            if mask.shape != (bsz, kv_len):
                raise ValueError(
                    f"mask shape must be (batch, kv_len)=({bsz}, {kv_len}), "
                    f"got {tuple(mask.shape)}"
                )
            mask = mask.to(dtype=k.dtype, device=k.device).unsqueeze(1).unsqueeze(-1)
            k = k * mask
            v = v * mask

        # kv_summary: (B, H, D, D)
        kv_summary = torch.einsum("bhld,bhle->bhde", k, v)
        # k_sum: (B, H, D)
        k_sum = k.sum(dim=2)

        # numerator: (B, H, Q, D)
        numerator = torch.einsum("bhqd,bhde->bhqe", q, kv_summary)
        # denominator: (B, H, Q, 1)
        denominator = torch.einsum("bhqd,bhd->bhq", q, k_sum).unsqueeze(-1)
        out = numerator / (denominator + self.eps)

        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out
