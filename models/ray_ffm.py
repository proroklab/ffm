import math
from typing import Any, Dict, List, Optional, Tuple

import gym
import torch
import numpy as np

from popgym.baselines.ray_models.base_model import BaseModel
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import wandb

from .ffm_outer import FFM
from .ffm_outer_double import FFMDouble


class LoggingCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        model = algorithm.get_policy().model
        if isinstance(model, RayFFM):
            core = model.core
            if isinstance(core, FFM):
                a = core.ffa.a.detach().cpu()
                b = core.ffa.b.detach().cpu()

                a_output = a.flatten().clamp_(max=1e-8)
                decay_period = (math.log(0.01) / a_output).clamp(max=1024)
                periods = (2 * math.pi / b.flatten().abs()).clamp_(max=1024)

                result["conv_a"] = wandb.Histogram(a_output, num_bins=a.numel())

                for i, v in enumerate(a_output):
                    result[f"agg_a{i}"] = a_output[i].item()
                for i, v in enumerate(periods):
                    result[f"agg_w{i}"] = periods[i].item()

                try:
                    result["conv_a_time_to_1pct"] = wandb.Histogram(decay_period)
                except ValueError:
                    pass
                result["conv_b_period"] = wandb.Histogram(periods.flatten())
                result["y_mean"] = core._y.mean().cpu().item()
                result["y_max"] = core._y.max().cpu().item()
                result["y_sparsity"] = (core._y > 0.01).float().mean().cpu().item()
                result["state_mean"] = core._state.mean().cpu().item()
                result["state_max"] = core._state.max().cpu().item()
                if hasattr(core, '_p'):
                    result["input_gate_mean"] = core._p.mean().cpu().item()
                    result["input_gate_max"] = core._p.max().cpu().item()
                    result["input_gate_std"] = core._p.std().cpu().item()
                    result["output_gate_mean"] = core._q.mean().cpu().item()
                    result["output_gate_max"] = core._q.max().cpu().item()
                    result["output_gate_std"] = core._q.std().cpu().item()
                if hasattr(core, '_out'):
                    result["out_mean"] = core._out.mean().cpu().item()
                    result["out_max"] = core._out.max().cpu().item()
                    result["out_std"] = core._out.std().cpu().item()
                    result["output_gate_std"] = core._out.std().cpu().item()



class RayFFM(BaseModel):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 32,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {},
    }
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        **custom_model_kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.core = FFM(
            input_size=self.cfg["preprocessor_output_size"],
            hidden_size=self.cfg["hidden_size"],
            memory_size=self.cfg["memory_size"],
            output_size=self.cfg["hidden_size"],
            context_size=self.cfg["context_size"],
            stats=self.cfg["record_stats"],
            **self.cfg["agg_kwargs"],
        )
        print(self.core)

    def initial_state(self) -> List[torch.Tensor]:
        return [
            # Last dim is real and imag
            torch.zeros(1, self.cfg["memory_size"], self.cfg["context_size"] // 2, 2)
            for i in range(self.core.num_states)
        ]

    def forward_memory(
        self,
        z: torch.Tensor,
        state: List[torch.Tensor],
        t_starts: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        state = [torch.view_as_complex(s) for s in state]
        z, state = self.core(z, state)
        state = [torch.view_as_real(s) for s in state]
        return z, state


class RayFFMDouble(BaseModel):
    MODEL_CONFIG = {
        "hidden_size": 128,
        "memory_size": 8,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {},
    }
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        **custom_model_kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.core = FFMDouble(
            input_size=self.cfg["preprocessor_output_size"],
            hidden_size=self.cfg["hidden_size"],
            memory_size=self.cfg["memory_size"],
            output_size=self.cfg["hidden_size"],
            context_size=self.cfg["context_size"],
            stats=self.cfg["record_stats"],
            **self.cfg["agg_kwargs"],
        )
        print(self.core)

    def initial_state(self) -> List[torch.Tensor]:
        return [
            # Last dim is real and imag
            torch.zeros(1, self.cfg["memory_size"], self.cfg["context_size"] // 2, 2)
            for i in range(self.core.num_states)
        ]

    def forward_memory(
        self,
        z: torch.Tensor,
        state: List[torch.Tensor],
        t_starts: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        state = [torch.view_as_complex(s) for s in state]
        z, state = self.core(z, state)
        state = [torch.view_as_real(s) for s in state]
        return z, state

def norm(x):
    return x / x.logsumexp(dim=-1, keepdim=True)

def maxnorm(x):
    return x / x.abs().max(dim=-1, keepdim=True).values

class RayFFMTanh(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 32,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {"input_act": torch.nn.Softsign()},
    }

class RayFFMSquare(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 16,
        "context_size": 16,
        "record_stats": True,
        "agg_kwargs": {"input_act": torch.nn.Softsign()},
    }

class RayFFMNoDecay(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 32,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {"decay": False},
    }

class RayFFMNoLearnDecay(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 32,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {"learn_decay": False},
    }

class RayFFMNoOscillate(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 32,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {"oscillate": False},
    }

class RayFFMNoLearnOscillate(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 32,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {"learn_oscillate": False},
    }

class RayFFM1024(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 64,
        "context_size": 16,
        "record_stats": True,
        "agg_kwargs": {},
    }

class RayFFM2048(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 64,
        "context_size": 32,
        "record_stats": True,
        "agg_kwargs": {},
    }

class RayFFM4096(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 128,
        "context_size": 32,
        "record_stats": True,
        "agg_kwargs": {},
    }


class RayFFMNoInGate(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 32,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {"input_gate": False},
    }



class RayFFMNoOutGate(RayFFM):
    MODEL_CONFIG = {
        "hidden_size": 256,
        "memory_size": 32,
        "context_size": 8,
        "record_stats": True,
        "agg_kwargs": {"output_gate": False},
    }
