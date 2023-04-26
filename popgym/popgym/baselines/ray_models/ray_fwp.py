import math
from typing import List, Tuple

import gym
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from popgym.baselines.models.fwp import FWPBlock
from popgym.baselines.ray_models.base_model import BaseModel


class FastWeightProgrammer(BaseModel):
    r"""The fast weight programmer from

    @inproceedings{schlag_linear_2021,
        title = {Linear {Transformers} {Are} {Secretly} {Fast} {Weight} {Programmers}},
        url = {https://proceedings.mlr.press/v139/schlag21a.html},
        language = {en},
        urldate = {2022-09-21},
        booktitle = {
            Proceedings of the 38th International Conference on Machine Learning
        },
        publisher = {PMLR},
        author = {Schlag, Imanol and Irie, Kazuki and Schmidhuber, Jürgen},
        month = jul,
        year = {2021},
        note = {ISSN: 2640-3498},
        pages = {9355--9366},
    }

    without the RNN extensions."""

    MODEL_CONFIG = {
        # Whether to use the sum normalization over the key/query term
        # as in the paper
        "sum_normalization": True,
        # Which positional embedding to use
        "embedding": "sine",
        # Which cumulative aggregator to use. Only sum is used in the paper.
        # This can be sum or max
        "aggregator": "sum",
        # Whether the recurrent state size should be hidden_size
        # or hidden_size ** 2
        "big_hidden": False,
    }

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        if self.cfg["big_hidden"]:
            self.h = self.cfg["hidden_size"]
        else:
            self.h = round(math.sqrt(self.cfg["hidden_size"]))
        self.core = FWPBlock(
            input_size=self.cfg["preprocessor_output_size"],
            hidden_size=self.h,
            aggregator=self.cfg["aggregator"],
            sum_normalization=self.cfg["sum_normalization"],
        )
        if not self.cfg["big_hidden"]:
            self.unmap = nn.Linear(self.h, self.cfg["hidden_size"])

    def initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(1, self.h, self.h),
        ]

    def forward_memory(
        self,
        z: TensorType,
        state: List[TensorType],
        t_starts: TensorType,
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        B, T, _ = z.shape
        [memory] = state
        z, memory = self.core(z, memory)
        h = z.shape[-1]
        if not self.cfg["big_hidden"]:
            z = self.unmap(z)
        return z, [
            memory[:, -1].reshape(B, 1, h, h),
        ]


class BigFastWeightProgrammer(FastWeightProgrammer):
    MODEL_CONFIG = {
        # Whether to use the sum normalization over the key/query term
        # as in the paper
        "sum_normalization": True,
        # Which positional embedding to use
        "embedding": "sine",
        # Which cumulative aggregator to use. Only sum is used in the paper.
        # This can be sum or max
        "aggregator": "sum",
        # Whether the recurrent state size should be hidden_size
        # or hidden_size ** 2
        "big_hidden": True,
    }
