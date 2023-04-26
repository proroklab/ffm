import math
from typing import Any, Dict, List, Optional, Tuple

import gym
import torch
from opt_einsum import contract
from aggregations import LaplaceAggregation, OuterLaplaceAggregation, MaxAggregation, SumAggregation, ProdAggregation
from torch import nn


class FFM(nn.Module):
    """The standard FFM cell from the paper"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int,
        output_size: int,
        context_size: int,
        stats: bool = False,
        input_gate: bool = True,
        input_act: Optional[callable] = None,
        output_gate: bool = True,
        **agg_kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.context_size = context_size
        self.stats = stats
        self.num_states = 1
        self.input_gate = input_gate
        self.output_gate = output_gate

        self.pre = nn.Linear(input_size, 2 * memory_size + 2 * output_size)
        self.input_act = input_act
        self.ffa = OuterLaplaceAggregation(
            d_model=memory_size, 
            context_size=context_size // 2, 
            max_len=1024,
            min_period=1,
            max_period=1024,
            **agg_kwargs
        )
        self.mix = nn.Linear(memory_size * context_size, output_size)
        self.ln_out = nn.LayerNorm(output_size, elementwise_affine=False)

    def forward(
        self, 
        x: torch.Tensor, 
        states: Tuple[Optional[torch.Tensor]] = (None,)
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        B, T, F = x.shape
        [state] = states
        if state is None:
            state = torch.zeros(
                (self.memory_size, self.context_size),
                device=x.device, dtype=torch.complex64
            )

        # Compute values from x
        y, thru, gate = self.pre(x).split([
            self.memory_size, self.output_size, self.memory_size + self.output_size
        ], dim=-1)

        # Compute gates
        gate = gate.sigmoid()
        in_gate, out_gate = gate.split([
            self.memory_size, self.output_size
        ], dim=-1)

        if self.input_act:
            y = self.input_act(y)
        if self.input_gate:
            y = y * in_gate

        # Compute state and output
        state = self.ffa(y.unsqueeze(-1), state) # Last dim for context
        z = self.mix(torch.view_as_real(state).flatten(2))

        if self.output_gate:
            out = self.ln_out(z) * out_gate + thru * (1 - out_gate)
        else:
            out = self.ln_out(z) + thru

        if self.stats and self.training:
            self._y = y.detach()
            self._state = z.detach()
            self._q = out_gate.detach()
            self._p = in_gate.detach()
            self._out = out.detach()

        return out, [state]
