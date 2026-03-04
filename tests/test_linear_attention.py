import torch

from kimi_attention import KimiLinearAttention


def test_forward_shape_self_attention() -> None:
    torch.manual_seed(0)
    model = KimiLinearAttention(dim=64, num_heads=8)
    x = torch.randn(2, 16, 64)

    out = model(x)

    assert out.shape == (2, 16, 64)


def test_forward_cross_attention_with_mask() -> None:
    torch.manual_seed(0)
    model = KimiLinearAttention(dim=32, num_heads=4, dropout=0.0)
    q = torch.randn(3, 5, 32)
    kv = torch.randn(3, 7, 32)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    out = model(q, context=kv, mask=mask)

    assert out.shape == (3, 5, 32)
    assert torch.isfinite(out).all()


def test_backward_pass() -> None:
    torch.manual_seed(42)
    model = KimiLinearAttention(dim=48, num_heads=6)
    x = torch.randn(2, 10, 48, requires_grad=True)

    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
