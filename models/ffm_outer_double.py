import math
from typing import Any, Dict, List, Optional, Tuple

import gym
import torch
from opt_einsum import contract
from aggregations import LaplaceAggregation, OuterLaplaceAggregation, MaxAggregation, SumAggregation, ProdAggregation
from torch import nn


class FFMDouble(nn.Module):
    """A block of two temporal convolutions"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        output_size: int,
        context_size: int,
        stats: bool = False,
        bias: bool = False,
        **agg_kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.context_size = context_size
        self.stats = stats
        self.num_states = 2

        self.pre0 = nn.Linear(input_size, 2 * memory_size + 2 * hidden_size)
        self.pre1 = nn.Linear(hidden_size, 2 * memory_size + 2 * output_size)
        self.ffa0 = OuterLaplaceAggregation(
            d_model=memory_size, 
            context_size=context_size // 2, 
            max_len=1024,
            min_period=1,
            max_period=1024,
            **agg_kwargs
        )
        self.ffa1 = OuterLaplaceAggregation(
            d_model=memory_size, 
            context_size=context_size // 2, 
            max_len=1024,
            min_period=1,
            max_period=1024,
            **agg_kwargs
        )
        self.mix0 = nn.Linear(memory_size * context_size, hidden_size)
        self.mix1 = nn.Linear(memory_size * context_size, output_size)
        self.ln_out0 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_out1 = nn.LayerNorm(output_size, elementwise_affine=False)

    def forward(
        self, 
        x: torch.Tensor, 
        states: Tuple[torch.Tensor] = (None,)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        B, T, F = x.shape
        [state0, state1] = states
        if state0 is None:
            state0 = torch.zeros(
                (self.memory_size, self.context_size),
                device=x.device, dtype=torch.complex64
            )
        # Compute values from x
        y0, thru0, gate0 = self.pre0(x).split([
            self.memory_size, self.hidden_size, self.memory_size + self.hidden_size
        ], dim=-1)

        # Compute gates
        gate0 = gate0.sigmoid()
        in_gate0, out_gate0 = gate0.split([
            self.memory_size, self.hidden_size
        ], dim=-1)


        # Compute state and output
        y0 = y0 * in_gate0
        state0 = self.ffa0(y0.unsqueeze(-1), state0) # Last dim for context
        z0 = self.mix0(torch.view_as_real(state0).flatten(2))
        out0 = self.ln_out0(z0) * out_gate0 + thru0 * (1 - out_gate0)

        # Second layer
        y1, thru1, gate1 = self.pre1(out0).split([
            self.memory_size, self.hidden_size, self.memory_size + self.hidden_size
        ], dim=-1)

        # Compute gates
        gate1 = gate1.sigmoid()
        in_gate1, out_gate1 = gate1.split([
            self.memory_size, self.hidden_size
        ], dim=-1)


        # Compute state and output
        y1 = y1 * in_gate1
        state1 = self.ffa1(y1.unsqueeze(-1), state1) # Last dim for context
        z1 = self.mix1(torch.view_as_real(state1).flatten(2))
        out1 = self.ln_out1(z1) * out_gate1 + thru1 * (1 - out_gate1)

        return out1, [state0, state1]
