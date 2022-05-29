from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
import torch.nn as nn
from pelutils import DataStorage


@dataclass
class ModelConfig(DataStorage, json_name="model_config.json", indent=4):
    state_size: int
    hidden_layer_sizes: list[int]
    num_residual_blocks: int
    residual_size: int
    dropout: float


class _BaseModel(abc.ABC, nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(nn.Module, self).__init__()
        self.cfg = cfg
        self.buid_model()

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def numel(self) -> int:
        """ Number of model parameters. Further docs here: https://pokemondb.net/pokedex/numel """
        return sum(p.numel() for p in self.parameters())

    def activation_transform_layers(self, size: int) -> tuple[nn.Module]:
        return (
            nn.ReLU(),
            nn.BatchNorm1d(size),
            nn.Dropout(p=self.cfg.dropout),
        )

class Model(_BaseModel):

    def build_model(self):

        # Build initial fully connected layers
        fully_connected = list()
        layer_sizes = [self.cfg.state_size] + self.cfg.hidden_layer_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            fully_connected.extend([
                nn.Linear(in_size, out_size),
                *self.activation_transform_layers(out_size),
            ])
        self.fully_connected = nn.Sequential(*fully_connected)

        # Build residual layers
        self.residual_blocks = nn.Sequential(*(
            _ResidualBlock(self.cfg) for _ in range(self.cfg.num_residual_blocks)
        ))

        # Final linear output layer
        self.output_layer = nn.Linear(self.cfg.residual_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fully_connected(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x

class _ResidualBlock(_BaseModel):

    num_layers = 2

    def build_model(self):
        fully_connected = list()
        for i in range(self.num_layers):
            fully_connected.append(
                nn.Linear(self.cfg.residual_size, self.cfg.residual_size)
            )
            if i < self.num_layers - 1:
                fully_connected.append(
                    self.activation_transform_layers(self.cfg.residual_size)
                )
        self.fully_connected = nn.Sequential(*fully_connected)

        self.output_transform = self.activation_transform_layers(self.cfg.residual_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.fully_connected(x)
        x = fx + x
        x = self.output_transform(x)
        return x

