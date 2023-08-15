from typing import Callable, Optional, Tuple
from ffm.ffa import FFA, LogspaceFFA
import jax
#jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import equinox as eqx
from equinox import nn
from functools import partial
import torch

from ffm_jax import ffa

if __name__ == '__main__':
    f = FFA(2, 3)
    L = 1024
    params = (
        jnp.array(f.a.detach().numpy()).reshape(-1),
        jnp.array(f.b.detach().numpy()).reshape(-1)
    )
    # torch
    x = torch.arange(2 * L, dtype=torch.float32).reshape(1, L, 2)
    state = torch.rand(1, 1, 2, 3, dtype=torch.complex64)
    out = f(x, state)

    # jax
    x_j = x.detach().numpy().reshape(L, 2) 
    state_j = state.detach().numpy().reshape(1, 2, 3)
    done_j = jnp.zeros(L, dtype=bool)
    out_j = ffa.apply(params, x_j, state_j, done_j)

    diff = jnp.abs(out_j - out.detach().numpy())
    diff = jnp.sum(diff, axis=-1)
    print(jnp.max(diff), jnp.argmax(diff))
    assert jnp.all(diff < 0.3)