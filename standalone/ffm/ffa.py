import math
import torch
from torch import nn


class FFA(nn.Module):
    def __init__(
        self,
        # Required
        memory_size: int,
        context_size: int,
        # Model Settings
        max_len: int = 1024,
        dtype: torch.dtype = torch.double,
        oscillate: bool = True,
        learn_oscillate: bool = True,
        decay: bool = True,
        learn_decay: bool = True,
        fudge_factor: float = 0.01,
        # Weight Init Settings
        min_period: int = 1,
        max_period: int = 1024,
        forgotten_at: float = 0.01,
        modify_real: bool = True,
    ):
        """A phaser-encoded aggregation operator
        Inputs:
            Required Settings:
                memory_size: Feature dimension of the model

            Model Settings:
                max_len: Maximum length of the batch in timesteps. Note this
                    is not the episode length, but rather the sequence length. The model
                    will be fastest if all sequences are within max_len. But we may
                    experience floating point under/overflow for very long sequences.
                    Setting this to less than the sequence length will break the
                    sequence into parts, trading off speed for accuracy. Gradients
                    will propagate as usual across the boundaries.
                context_size: The number of context filters for each channel
                    of memory_size.
                dtype: Whether to use floats or doubles. Note doubles enables
                    significantly more representational power for a little
                    extra compute.
                oscillate: Whether we should use the imaginary component of
                    the exponential (sinusoidal oscillations). If this is false,
                    the model cannot determine relative time between inputs.
                learn_oscillate: Whether the oscillate terms (omega in the paper)
                    should be learned.
                decay: Whether we should use the real component of the exponential.
                    If this is false, the model cannot decay inputs over time.
                learn_decay: Whether the decay terms (alpha in the paper)
                    should be learned.
                fudge_factor: A small positive number to prevent floating point
                    overflows.

            Weight Initialization Settings:
                forgetten_at: What fraction of the original input a memory is considered
                    "forgotten" at.
                min_period: The initial minimum sinusoidal period and trace decay. 
                    This is the minimum relative time distance the model can 
                    initially represent. Note that if min/max period are learned,
                    they can exceed the limits set here.
                max_period: The initial maximum sinusoidal period and trace decay. This 
                    determines the maximum relative time distance the model can initially
                    represent.
                modify_real: If this is false, min_period, max_period, and forgotten_at
                    will not affect the alpha term initialization.
        """
        super().__init__()
        self.memory_size = memory_size
        self.max_len = max_len
        self.context_size = context_size
        self.oscillate = oscillate
        self.learn_oscillate = learn_oscillate
        self.decay = decay
        self.learn_decay = learn_decay
        assert dtype in [torch.float, torch.double]
        self.dtype = dtype
        dtype_max = torch.finfo(dtype).max
        # To prevent overflows, ensure exp(limit * max_len) < {float,double}
        # limit * max_len < log({float,double})
        # limit == log({float,double}) / max_len - fudge_factor
        self.limit = math.log(dtype_max) / max_len - fudge_factor

        # Memories will be a fraction (epsilon) of their original value
        # at max_period
        # exp(a * max_period) < epsilon
        # a = < log(epsilon) / max_period
        if modify_real:
            soft_high = math.log(forgotten_at) / max_period
        else:
            soft_high = -1e-6

        # Initialize parameters
        real_param_shape = [1, 1, self.memory_size]
        imag_param_shape = [1, 1, self.context_size]
        a_low = -self.limit + fudge_factor
        a_high = max(min(-1e-6, soft_high), a_low)

        a = torch.linspace(a_low, a_high, real_param_shape[-1]).reshape(
            real_param_shape
        )
        b = (
            2
            * torch.pi
            / torch.linspace(min_period, max_period, imag_param_shape[-1]).reshape(
                imag_param_shape
            )
        )

        if not self.oscillate:
            b.fill_(1 / 1e6)
        if not self.decay:
            a.fill_(0)

        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)

        # For typechecker
        self.one: torch.Tensor
        self.inner_idx: torch.Tensor
        self.outer_idx: torch.Tensor
        self.state_offset: torch.Tensor
        # Buffers
        self.register_buffer("one", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("inner_idx", torch.arange(max_len, dtype=dtype).flip(0))
        self.register_buffer("outer_idx", -self.inner_idx)
        self.register_buffer("state_offset", torch.arange(1, max_len + 1, dtype=dtype))

    def extra_repr(self):
        return f"in_features={self.memory_size}, out_features=({self.memory_size}, {self.context_size})"

    def gamma(self, t_minus_i: torch.Tensor) -> torch.Tensor:
        """Gamma function from the paper

        Args:
            t_minus_i: 1D tensor of relative time indices (can be discrete or cont.)

        Returns:
            gamma^t for t in t_minus_i
        """
        assert t_minus_i.dim() == 1
        T = t_minus_i.shape[0]
        self.a.data = self.a.data.clamp(min=-self.limit, max=-1e-8)
        a = self.a.clamp(min=-self.limit, max=-1e-8)
        b = self.b
        if not self.oscillate or not self.learn_oscillate:
            b = b.detach()
        if not self.decay or not self.learn_decay:
            a = a.detach()

        exp = torch.complex(
            a.reshape(1, 1, -1, 1),
            b.reshape(1, 1, 1, -1),
        )
        out = torch.exp(exp * t_minus_i.reshape(1, T, 1, 1))
        return out

    def batched_recurrent_update(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Given x_{k:n} and s_{k-1}, compute s{k:n}

        Args:
            x: input of [B, T, memory_size]
            memory: state of [B, 1, memory_size, context_size]

        Returns
            state of [B, n-k, memory_size, context_size]
        """
        B, T, F, D = x.shape
        z = torch.cumsum(self.gamma(self.inner_idx[:T]) * x, dim=1)
        memory = self.gamma(self.outer_idx[:T]) * z + memory * self.gamma(
            self.state_offset[:T]
        )

        return memory.to(torch.complex64)

    def single_step_update(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """A fast recurrent update for a single timestep"""
        return x + memory * self.gamma(self.one)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: [B, T, F, 1]
            memory: [B, 1, F, dtype=torch.complex]
        Returns:
            memory: [B, 1, F, dtype=torch.complex]
        """
        assert memory.dtype in [
            torch.complex64,
            torch.complex128,
        ], "State should be complex dtype"
        assert x.dim() == 3
        assert memory.dim() == 4

        B, T, F = x.shape
        x = x.reshape(B, T, F, 1)

        if x.shape[1] == 1:
            # More efficient shortcut for single-timestep inference
            return self.single_step_update(x, memory)
        elif x.shape[1] < self.max_len:
            # Default case, the whole thing can fit into a single temporal batch
            return self.batched_recurrent_update(x, memory)
        else:
            # Need to break into temporal batches
            chunks = x.split(self.max_len, dim=1)
            states = []
            for chunk in chunks:
                memory = self.batched_recurrent_update(chunk, memory[:, -1:])
                states.append(memory)
            return torch.cat(states, dim=1)
