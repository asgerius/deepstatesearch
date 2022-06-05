from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pelutils import log, thousands_seperators, TT
from pelutils.datastorage import DataStorage
from pelutils.parser import JobDescription
from deepspeedcube import tensor_size

from deepspeedcube.model import Model, ModelConfig
from deepspeedcube.envs import get_env
from deepspeedcube.envs.gen_states import gen_new_states
from deepspeedcube.model.generator_network import clone_model, update_generator_network


@dataclass
class TrainConfig(DataStorage, json_name="train_config.json", indent=4):
    env: str
    num_models: int
    batches: int
    batch_size: int
    scramble_depth: int
    lr: float
    tau: float
    j_norm: float

@dataclass
class TrainResults(DataStorage, json_name="train_results.json", indent=4):
    eval_idx: list[int]
    # depth [ batch [ value ] ]
    value_estimations: list[list[float]] = field(default_factory=list)

    lr: list[float] = field(default_factory=list)
    # model [ batch [ loss ] ]
    losses: list[list[float]] = field(default_factory=list)

def train(job: JobDescription):

    # Args either go into the model config or into the training config
    # Those that go into the training config are filtered out here
    train_cfg = TrainConfig(
        env            = job.env,
        num_models     = job.num_models,
        batches        = job.batches,
        batch_size     = job.batch_size,
        scramble_depth = job.scramble_depth,
        lr             = job.lr,
        tau            = job.tau,
        j_norm         = job.j_norm,
    )
    log("Got training config", train_cfg)

    log("Setting up environment '%s'" % train_cfg.env)
    env = get_env(train_cfg.env)

    eval_batches = np.arange(0, train_cfg.batches, 500, dtype=int).tolist()
    if (last_batch_idx := train_cfg.batches - 1) != eval_batches[-1]:
        eval_batches.append(last_batch_idx)
    log("Evaluating at batches", eval_batches)

    log.section("Building models")
    model_cfg = ModelConfig(
        state_size          = env.state_oh_size,
        hidden_layer_sizes  = job.hidden_layer_sizes,
        num_residual_blocks = job.num_residual_blocks,
        residual_size       = job.residual_size,
        dropout             = job.dropout,
    )
    log("Got model config", model_cfg)

    criterion = nn.MSELoss()
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
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20000)
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
    log(models[0], "Size in bytes: %s" % thousands_seperators(4*models[0].numel()))

    log.section("Starting training")
    train_results = TrainResults(eval_idx=eval_batches)
    train_results.value_estimations.extend(list() for _ in range(train_cfg.scramble_depth))
    train_results.losses.extend(list() for _ in range(train_cfg.num_models))

    for i in range(train_cfg.batches):

        if i in eval_batches:
            TT.profile("Evaluate")
            log("Evaluating models")

            states, depths = gen_new_states(env, 10 ** 4, train_cfg.scramble_depth)
            states_oh = env.multiple_oh(states)
            with TT.profile("Value estimates"), torch.no_grad():
                preds = torch.zeros(len(states))
                for model in models:
                    preds += model(states_oh).squeeze()
                preds = train_cfg.j_norm * preds / len(models)

                for j in range(train_cfg.scramble_depth):
                    train_results.value_estimations[j].append(
                        preds[depths == j + 1].mean().item()
                    )

            TT.end_profile()

        TT.profile("Batch")
        log("Batch %i / %i" % (i+1, train_cfg.batches))

        log.debug("Generating %i states" % (train_cfg.batch_size * train_cfg.num_models))
        with TT.profile("Generate scrambled states"):
            all_states, _ = gen_new_states(
                env,
                train_cfg.batch_size * train_cfg.num_models,
                train_cfg.scramble_depth,
            )
        log.debug("Size of all states in bytes: %s" % thousands_seperators(tensor_size(all_states)))

        for j in range(train_cfg.num_models):
            log.debug("Training model %i / %i" % (j+1, train_cfg.num_models))

            model = models[j]
            gen_model = gen_models[j]
            optimizer = optimizers[j]
            scheduler = schedulers[j]

            log.debug("Generating neighbour states")
            states = all_states[j*train_cfg.batch_size:(j+1)*train_cfg.batch_size]
            neighbour_states = env.neighbours(states)

            log.debug("Forward passing states")
            with TT.profile("OH neighbour states"):
                neighbour_states_oh = env.multiple_oh(neighbour_states)
            assert neighbour_states.is_contiguous() and neighbour_states_oh.is_contiguous()

            with TT.profile("Value estimates"), torch.no_grad():
                # TODO Forward pass only unique and non-solved states
                J = gen_model(neighbour_states_oh).squeeze()

            with TT.profile("Set solved states to j = 0"):
                solved_states = env.multiple_is_solved(neighbour_states)
                J[solved_states] = 0
            J = J.view(len(states), len(env.action_space))

            with TT.profile("Calculate targets"):
                g = 1
                targets = torch.min(g + J, dim=1).values

            with TT.profile("OH states"):
                states_oh = env.multiple_oh(states)
            assert states.is_contiguous() and states_oh.is_contiguous()

            with TT.profile("Train model"):
                preds = model(states_oh).squeeze()
                loss = criterion(preds, targets)
                loss.backward()
                log.debug("Loss: %.4f" % loss.item())
                train_results.losses[j].append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            if j == 0:
                train_results.lr.append(schedulers[j].get_last_lr()[0])
                log.debug(
                    "Shapes",
                    "states:              %s" % list(states.shape),
                    "states_oh:           %s" % list(states_oh.shape),
                    "neighbour_states:    %s" % list(neighbour_states.shape),
                    "neighbour_states_oh: %s" % list(neighbour_states_oh.shape),
                    "value_estimates:     %s" % list(J.shape),
                    "targets:             %s" % list(targets.shape),
                    sep="\n    ",
                )
                log.debug(
                    "Sizes in bytes",
                    "states:              %s" % thousands_seperators(tensor_size(states)),
                    "states_oh:           %s" % thousands_seperators(tensor_size(states_oh)),
                    "neighbour_states:    %s" % thousands_seperators(tensor_size(neighbour_states)),
                    "neighbour_states_oh: %s" % thousands_seperators(tensor_size(neighbour_states_oh)),
                    "value_estimates:     %s" % thousands_seperators(tensor_size(J)),
                    "targets:             %s" % thousands_seperators(tensor_size(targets)),
                    sep="\n    ",
                )

            scheduler.step()

            with TT.profile("Update generator network"):
                update_generator_network(train_cfg.tau, gen_models[j], models[j])

        log("Mean loss: %.4f" % (sum(ls[-1] for ls in train_results.losses) / train_cfg.num_models))

        TT.end_profile()

    log.section("Saving")
    with TT.profile("Save"):
        train_cfg.save(job.location)
        model_cfg.save(job.location)
        train_results.save(job.location)
