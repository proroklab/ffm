from typing import Optional, Tuple
import torch
from torch import nn
from .ffa import FFA, LogspaceFFA


class FFM(nn.Module):
    """Fast and Forgetful Memory

    Args:
        input_size: Size of input features to the model
        hidden_size: Size of hidden layers within the model
        memory_size: Size of the decay dimension of memory (m in the paper)
        context_size: Size of the temporal context (c in the paper, the
            total recurrent size is m * c)
        output_size: Output feature size of the model
        min_period: Minimum period for FFA, see FFA for details
        max_period: Maximum period for FFA, see FFA for details

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        context_size: int,
        output_size: int,
        min_period: int = 1,
        max_period: int = 1024,
        logspace: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.context_size = context_size

        self.pre = nn.Linear(input_size, 2 * memory_size + 2 * output_size)
        if logspace:
            self.ffa = LogspaceFFA(
                memory_size=memory_size,
                context_size=context_size,
                min_period=min_period,
                max_period=max_period,
            )
        else:
            self.ffa = FFA(
                memory_size=memory_size,
                context_size=context_size,
                min_period=min_period,
                max_period=max_period,
            )
        self.mix = nn.Linear(2 * memory_size * context_size, output_size)
        self.ln_out = nn.LayerNorm(output_size, elementwise_affine=False)

    def initial_state(self, batch_size=1, device='cpu'):
        return torch.zeros(
            (batch_size, 1, self.memory_size, self.context_size),
            device=device,
            dtype=torch.complex64,
        )


    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input of shape [B, T, input_size]
            state: Recurrent state of size [B, 1, memory_size, context_size]
                and dtype=complex

        Returns:
            y: Output of shape [B, T, output_size]
            state: Recurrent state of size [B, 1, memory_size, context_size]
                and dtype=complex
        """

        B, T, _ = x.shape
        if state is None:
            # Typechecker doesn't like us to reuse the 'state' name
            s = self.initial_state(B, x.device)
        else:
            s = state

        # Compute values from x
        y, thru, gate = self.pre(x).split(
            [self.memory_size, self.output_size, self.memory_size + self.output_size],
            dim=-1,
        )

        # Compute gates
        gate = gate.sigmoid()
        in_gate, out_gate = gate.split([self.memory_size, self.output_size], dim=-1)

        # Compute state and output
        s = self.ffa((y * in_gate), s)  # Last dim for context
        z = self.mix(torch.view_as_real(s).reshape(B, T, -1))
        out = self.ln_out(z) * out_gate + thru * (1 - out_gate)

        return out, s


class DropInFFM(FFM):
    """Fast and Forgetful Memory, wrapped to behave like an nn.GRU

    Args:
        input_size: Size of input features to the model
        hidden_size: Size of hidden layers within the model
        memory_size: Size of the decay dimension of memory (m in the paper)
        context_size: Size of the temporal context (c in the paper, the
            total recurrent size is m * c)
        output_size: Output feature size of the model
        min_period: Minimum period for FFA, see FFA for details
        max_period: Maximum period for FFA, see FFA for details
        batch_first: Whether inputs/outputs/states are shape
            [batch, time, *]. If false, the inputs/outputs/states are
            shape [time, batch, *]

    """
    def __init__(self, *args, batch_first=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input of shape [B, T, F] or [B, F] if batch first,
                otherwise [T, B, F] or [T, F]
            state: Recurrent state of size [B, 1, M, C] or [B, M, C]
                and dtype=complex if batch_first, else
                [1, B, M, C] or [B, M, C]

        Returns:
            y: Output with the same batch dimensions as the input
            state: Recurrent state of the same shape as the input recurrent state
        """
        # Check if x missing singleton time or batch dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_y = True
        else:
            squeeze_y = False

        # Check if s is missing singleton time dim
        if self.batch_first:
            B, T, F = x.shape
        else:
            T, B, F = x.shape

        if state is None:
            # Typechecker doesn't like us to reuse the 'state' name
            s = self.initial_state(B, x.device)
        else:
            s = state

        s = s.reshape(B, 1, self.memory_size, self.context_size)

        # Sanity check shapes
        assert s.shape == (
            B,
            1,
            self.memory_size,
            self.context_size,
        ), (
            f"Given x of shape {x.shape}, expected state to be"
            f" shape [{B}, 1, {self.memory_size}, {self.context_size}], dtype=complex, but got {s.shape} "
            f"and dtype={s.dtype}"
        )   


        # Make everything batch first for FFM
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        # Call FFM
        y, s = super().forward(x, s)

        if not self.batch_first:
            y = y.permute(1, 0, 2)

        if squeeze_y:
            y = y.reshape(B, -1)

        # Return only terminal state of s
        s = s[:, -1]

        return y, s
