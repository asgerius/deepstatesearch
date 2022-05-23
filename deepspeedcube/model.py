from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from pelutils import DataStorage


@dataclass
class ModelConfig(DataStorage, json_name="model_config.json", indent=4):
    state_size: int
    hidden_layer_sizes: list[int]
    dropout: float


class Model(nn.Module):

    def __init__(self, cfg: ModelConfig):

        self.cfg = cfg
