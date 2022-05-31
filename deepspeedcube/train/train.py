from __future__ import annotations

from dataclasses import dataclass

from pelutils import log, thousands_seperators, TT
from pelutils.datastorage import DataStorage
from pelutils.parser import JobDescription

from deepspeedcube.model import Model, ModelConfig
from deepspeedcube.envs import get_env


@dataclass
class TrainConfig(DataStorage, json_name="train_config.json", indent=4):
    env: str

def train(job: JobDescription):

    # Args either go into the model config or into the training config
    # Those that go into the training config are filtered out here
    train_cfg = TrainConfig(
        env = job.env,
    )
    log("Got training config", train_cfg)

    log("Setting up environment '%s'" % train_cfg.env)
    env = get_env(train_cfg.env)

    log("Building model")
    model_cfg = ModelConfig(
        state_size          = env.state_oh_size,
        hidden_layer_sizes  = job.hidden_layer_sizes,
        num_residual_blocks = job.num_residual_blocks,
        residual_size       = job.residual_size,
        dropout             = job.dropout,
    )
    log("Got model config", model_cfg)

    with TT.profile("Building model"):
        model = Model(model_cfg)
    log("Build model", "Number of parameters: %s" % thousands_seperators(model.numel()))
