from ffm import FFM, DropInFFM
import torch


def main():
    B = 3
    T = 5
    f = FFM(
        input_size=32, hidden_size=55, memory_size=16, context_size=4, output_size=48
    )

    x = torch.randn(B, T, 32)
    y, s = f(x)
    assert y.shape == (B, T, 48)
    assert s.shape == (B, T, 16, 4)
    assert s.dtype == torch.complex64

    # time first
    f = DropInFFM(
        input_size=32, hidden_size=55, memory_size=16, context_size=4, output_size=48, batch_first=False
    )

    x = torch.randn(T, B, 32)
    y, s = f(x)
    assert y.shape == (T, B, 48)
    assert s.shape == (B, 16, 4)
    assert s.dtype == torch.complex64

    y, s = f(x, s)
    assert y.shape == (T, B, 48)
    assert s.shape == (B, 16, 4)
    assert s.dtype == torch.complex64

    # batch first
    f = DropInFFM(
        input_size=32, hidden_size=55, memory_size=16, context_size=4, output_size=48, batch_first=True
    )

    x = torch.randn(B, T, 32)
    y, s = f(x)
    assert y.shape == (B, T, 48)
    assert s.shape == (B, 16, 4)
    assert s.dtype == torch.complex64

    y, s = f(x, s)
    assert y.shape == (B, T, 48)
    assert s.shape == (B, 16, 4)
    assert s.dtype == torch.complex64

if __name__ == "__main__":
    main()
