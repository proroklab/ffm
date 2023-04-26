import math
import numpy as np
from opt_einsum import contract

import torch
from torch import nn


def get_aggregator(name: str) -> nn.Module:
    assert name in [
        "sum",
        "max",
    ], "Invalid aggregator. Must be 'sum' or 'max'"
    return {
        "sum": SumAggregation,
        "max": MaxAggregation,
        "laplace": LaplaceAggregation,
        "outer": OuterLaplaceAggregation,
    }[name]


class Aggregation(nn.Module):
    """Aggregates (x_k ... x_t , s_k) into s_t"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()


class SumAggregation(Aggregation):
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if memory.is_complex():
            res = x.clamp(-1e10, 1e10).cumsum(dim=1)
            real = res + memory.real
            imag = res + memory.imag
            return torch.complex(real, imag)
        else:
            res = x.clamp(-1e10, 1e10).cumsum(dim=1) + memory
            return res

class ProdAggregation(Aggregation):
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        return x.sigmoid().cumprod(dim=1) * memory


class MaxAggregation(Aggregation):
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        if memory.is_complex():
            real = torch.maximum(x.cummax(dim=1).values, memory.real)
            imag = torch.minimum(x.cummin(dim=1).values, memory.imag)
            return torch.complex(real, imag)
        else:
            res = torch.maximum(x.cummax(dim=1).values, memory)
            return torch.complex(res, res)


class OuterLaplaceAggregation(Aggregation):
    def __init__(
        self,
        # Required
        d_model: int,
        # Model Settings
        max_len: int = 1024,
        context_size: int = 4,
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
        modify_real: bool =  True,
        init_method: str = "linspace",
    ):
        """A phaser-encoded aggregation operator
        Inputs:
            Required Settings:
                d_model: Feature dimension of the model

            Model Settings:
                max_len: Maximum length of the batch in timesteps. Note this
                    is not the episode length, but rather the sequence length. The model
                    will be fastest if all sequences are within max_len. But we may
                    experience floating point under/overflow for very long sequences.
                    Setting this to less than the sequence length will break the
                    sequence into two parts, trading off speed for accuracy. Gradients
                    will propagate as usual across the boundary.
                context_size: The number of filters for each channel of d_model.
                dtype: Whether to use floats or doubles. Note doubles enables
                    significantly more representational power for a little
                    extra compute.
                oscillate: Whether we should use the imaginary component of 
                    the exponential (sinusoidal oscillations). If this is false, 
                    the model cannot determine relative time between inputs.
                decay: Whether we should use the real component of the exponential.
                    If this is false, the model cannot decay inputs over time.
                fudge_factor: A small positive number to prevent floating point 
                    overflows.

            Weight Initialization Settings:
                min_period: The initial minimum sinusoidal period. This is the minimum
                    relative time distance the model can initially represent.
                max_period: The initial maximum sinusoidal period. This determines 
                    the maximum relative time distance the model can initially 
                    represent.
                forgetten_at: What fraction of the original input a memory is considered
                    "forgotten" at. 
                modify_real: If this is false, min_period, max_period, and forgotten_at
                    will not affect the real value initialization.
                init_method: Can be either "random" or "linspace"
        """
        super().__init__()
        self.d_model = d_model
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
        real_param_shape = [1, 1, self.d_model]
        imag_param_shape = [1, 1, self.context_size]
        a_low = -self.limit + fudge_factor
        a_high = max(min(-1e-6, soft_high), a_low)
        # Log uniform distribution for a
        # TODO: This distribution is incorrect
        # we want the distribution to be uniform in the time it takes to decay
        # a value to 1% its original
        if init_method == "random":
            a = -torch.empty(real_param_shape).uniform_(math.log(-a_high), math.log(-a_low)).exp()
            # 2 pi / uniform dist for b
            b = 2 * torch.pi / torch.empty(imag_param_shape).uniform_(min_period, max_period)
        elif init_method == "point_normal":
            a = -torch.empty(real_param_shape)#.normal_(a_high, 0.001)
            nn.init.trunc_normal_(a, a_high, 0.001, a_low, a_high)
            # 2 pi / uniform dist for b
            b = 2 * torch.pi / torch.empty(imag_param_shape).uniform_(min_period, max_period)

        elif init_method == "logspace":
            a = torch.from_numpy((a_low + a_high) - np.geomspace(a_low, a_high, real_param_shape[-1])).reshape(real_param_shape).float()
            b = 2 * torch.pi / torch.linspace(min_period, max_period, imag_param_shape[-1]).reshape(imag_param_shape)
        elif init_method == "logspace_reverse":
            a = -torch.linspace(math.log(-a_high), math.log(-a_low), real_param_shape[-1]).exp().reshape(real_param_shape)
            b = 2 * torch.pi / torch.linspace(min_period, max_period, imag_param_shape[-1]).reshape(imag_param_shape)
        elif init_method == "linspace":
            a = torch.linspace(a_low, a_high, real_param_shape[-1]).reshape(real_param_shape)
            b = 2 * torch.pi / torch.linspace(min_period, max_period, imag_param_shape[-1]).reshape(imag_param_shape)
            
        else:
            raise NotImplementedError(f"Invalid init method: {init_method}")

        if not self.oscillate:
            b.fill_(1 / 1e6)
        if not self.decay:
            a.fill_(0)

        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)

        # Buffers
        self.register_buffer("one", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("inner_idx", torch.arange(max_len, dtype=dtype).flip(0))
        self.register_buffer("outer_idx", -self.inner_idx)
        self.register_buffer("state_offset", torch.arange(1, max_len + 1, dtype=dtype))

    def extra_repr(self):
        return f"in_features={self.d_model}, out_features=({self.d_model}, {self.context_size})"


    def psi(self, t_minus_i: torch.Tensor) -> torch.Tensor:
        assert t_minus_i.dim() == 1
        T = t_minus_i.shape[0]
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
        """A recurrent update for a batch over time"""
        B, T, F, D = x.shape
        z = torch.cumsum(self.psi(self.inner_idx[:T]) * x, dim=1)
        memory = self.psi(self.outer_idx[:T]) * z + memory * self.psi(
            self.state_offset[:T]
        )

        return memory.to(torch.complex64)

    def single_step_update(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """A fast recurrent update for a single timestep"""
        return x + memory * self.psi(self.one)

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
        assert x.dim() == 4
        assert memory.dim() == 4

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


class LaplaceAggregation(Aggregation):
    def __init__(
        self,
        # Required
        d_model: int,
        # Model Settings
        max_len: int = 1024,
        context_size: int = 1,
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
        modify_real: bool =  True,
        intact_ratio: float = 0.0,
        init_method: str = "linspace",
    ):
        """A phaser-encoded aggregation operator
        Inputs:
            Required Settings:
                d_model: Feature dimension of the model

            Model Settings:
                max_len: Maximum length of the batch in timesteps. Note this
                    is not the episode length, but rather the sequence length. The model
                    will be fastest if all sequences are within max_len. But we may
                    experience floating point under/overflow for very long sequences.
                    Setting this to less than the sequence length will break the
                    sequence into two parts, trading off speed for accuracy. Gradients
                    will propagate as usual across the boundary.
                context_size: The number of filters for each channel of d_model.
                dtype: Whether to use floats or doubles. Note doubles enables
                    significantly more representational power for a little
                    extra compute.
                oscillate: Whether we should use the imaginary component of 
                    the exponential (sinusoidal oscillations). If this is false, 
                    the model cannot determine relative time between inputs.
                decay: Whether we should use the real component of the exponential.
                    If this is false, the model cannot decay inputs over time.
                fudge_factor: A small positive number to prevent floating point 
                    overflows.

            Weight Initialization Settings:
                min_period: The initial minimum sinusoidal period. This is the minimum
                    relative time distance the model can initially represent.
                max_period: The initial maximum sinusoidal period. This determines 
                    the maximum relative time distance the model can initially 
                    represent.
                forgetten_at: What fraction of the original input a memory is considered
                    "forgotten" at. 
                modify_real: If this is false, min_period, max_period, and forgotten_at
                    will not affect the real value initialization.
                intact_ratio: Ratio denoting what proportion of memories should be
                    left intact. I.e., how many dimensions are not subject to decay
                    or oscillations. Increase this if you require very long-term memory.
        """
        super().__init__()
        self.d_model = d_model
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
            soft_low = math.log(forgotten_at) / min_period
        else:
            soft_high = -1e-6
            soft_low = -self.limit

        # Initialize parameters
        param_shape = [1, 1, self.d_model, self.context_size]
        a_low = max(-self.limit + fudge_factor, soft_low)
        a_high = max(min(-1e-6, soft_high), a_low)
        # Log uniform distribution for a
        # TODO: This distribution is incorrect
        # we want the distribution to be uniform in the time it takes to decay
        # a value to 1% its original
        if init_method == "random":
            a = -torch.empty(param_shape).uniform_(math.log(-a_high), math.log(-a_low)).exp()
            # 2 pi / uniform dist for b
            b = 2 * torch.pi / torch.empty(param_shape).uniform_(min_period, max_period)
        elif init_method == "logspace":
            a = torch.from_numpy((a_low + a_high) - np.geomspace(a_low, a_high, param_shape[-2])).reshape(param_shape).float()
            b = 2 * torch.pi / torch.linspace(min_period, max_period, param_shape[-2]).reshape(param_shape)
        elif init_method == "logspace_reverse":
            a = -torch.linspace(math.log(-a_high), math.log(-a_low), param_shape[-2]).exp().reshape(param_shape)
            b = 2 * torch.pi / torch.linspace(min_period, max_period, param_shape[-2]).reshape(param_shape)
        elif init_method == "linspace":
            a = torch.linspace(a_low, a_high, param_shape[-2]).reshape(param_shape)
            b = 2 * torch.pi / torch.linspace(min_period, max_period, param_shape[-2]).reshape(param_shape)
        else:
            raise NotImplementedError(f"Invalid init method: {init_method}")

        # Initialize intact memory
        num_intact = round(self.d_model * intact_ratio)
        a[:,:,:num_intact] = -1e-6
        b[:,:,:num_intact] = 1 / 1e6

        if not self.oscillate:
            b.fill_(1 / 1e6)
        if not self.decay:
            a.fill_(0)

        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)

        # Buffers
        self.register_buffer("one", torch.tensor([1.0], dtype=torch.float))
        self.register_buffer("inner_idx", torch.arange(max_len, dtype=dtype).flip(0))
        self.register_buffer("outer_idx", -self.inner_idx)
        self.register_buffer("state_offset", torch.arange(1, max_len + 1, dtype=dtype))

    def psi(self, t_minus_i: torch.Tensor) -> torch.Tensor:
        assert t_minus_i.dim() == 1
        T = t_minus_i.shape[0]
        # Compute for all filters/fourier series terms
        # e^(t * (a + bi))
        a = self.a
        b = self.b
        if not self.oscillate or not self.learn_oscillate:
            b = b.detach()
        if not self.decay or not self.learn_decay:
            a = a.detach()

        self._clamped_ab = torch.complex(a, b) 
        return torch.exp(
            self._clamped_ab * t_minus_i.reshape(1, T, 1, 1)
        )

    def batched_recurrent_update(
        self, x: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """A recurrent update for a batch over time"""
        B, T, F, D = x.shape
        z = torch.cumsum(self.psi(self.inner_idx[:T]) * x, dim=1)
        memory = self.psi(self.outer_idx[:T]) * z + memory * self.psi(
            self.state_offset[:T]
        )

        return memory.to(torch.complex64)

    def single_step_update(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """A fast recurrent update for a single timestep"""
        return x + memory * self.psi(self.one)

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
        assert x.shape[-1] == memory.shape[-2]
        x = x.unsqueeze(-1)

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
