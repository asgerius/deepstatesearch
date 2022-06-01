from __future__ import annotations

from dataclasses import dataclass, field

import torch.optim as optim
from pelutils import log, thousands_seperators, TT
from pelutils.datastorage import DataStorage
from pelutils.parser import JobDescription

from deepspeedcube.model import Model, ModelConfig
from deepspeedcube.envs import get_env
from deepspeedcube.train.generator_network import clone_model, update_generator_network


@dataclass
class TrainConfig(DataStorage, json_name="train_config.json", indent=4):
    env: str
    num_models: int
    batches: int
    lr: float
    tau: float

@dataclass
class TrainResults(DataStorage, json_name="train_results.json", indent=4):
    lr: list[float] = field(default_factory=list)

def train(job: JobDescription):

    # Args either go into the model config or into the training config
    # Those that go into the training config are filtered out here
    train_cfg = TrainConfig(
        env        = job.env,
        num_models = job.num_models,
        batches    = job.batches,
        lr         = job.lr,
        tau        = job.tau,
    )
    log("Got training config", train_cfg)

    log("Setting up environment '%s'" % train_cfg.env)
    env = get_env(train_cfg.env)

    log.section("Building models")
    model_cfg = ModelConfig(
        state_size          = env.state_oh_size,
        hidden_layer_sizes  = job.hidden_layer_sizes,
        num_residual_blocks = job.num_residual_blocks,
        residual_size       = job.residual_size,
        dropout             = job.dropout,
    )
    log("Got model config", model_cfg)

    models: list[Model] = list()
    gen_models: list[Model] = list()
    optimizers = list()
    schedulers = list()
    for _ in range(train_cfg.num_models):
        TT.profile("Build model")
        model = Model(model_cfg)
        gen_model = Model(model_cfg)
        clone_model(model, gen_model)
        optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)
        models.append(model)
        gen_models.append(gen_model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        TT.end_profile()
    log(
        "Built %i models" % train_cfg.num_models,
        "Parameters per model: %s" % thousands_seperators(models[0].numel()),
        "Total parameters:     %s" % thousands_seperators(sum(m.numel() for m in models)),
    )

    log.section("Starting training")
    train_results = TrainResults()
    for i in range(train_cfg.batches):
        TT.profile("Batch")
        log("Batch %i / %i" % (i+1, train_cfg.batches))
        for j in range(train_cfg.num_models):
            if j == 0:
                train_results.lr.append(schedulers[j].get_last_lr()[0])

            schedulers[j].step()

            with TT.profile("Update generator network"):
                update_generator_network(train_cfg.tau, gen_models[j], models[j])
        TT.end_profile()

    log.section("Saving")
    with TT.profile("Save"):
        train_cfg.save(job.location)
        model_cfg.save(job.location)
        train_results.save(job.location)
