from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pelutils import log, thousands_seperators, TT
from pelutils.datastorage import DataStorage
from pelutils.parser import JobDescription

from deepspeedcube import device, tensor_size
from deepspeedcube.model import Model, ModelConfig
from deepspeedcube.envs import get_env
from deepspeedcube.envs.gen_states import gen_new_states, get_batches_per_gen
from deepspeedcube.model.generator_network import clone_model, update_generator_network


@dataclass
class TrainConfig(DataStorage, json_name="train_config.json", indent=4):
    env:             str
    num_models:      int
    batches:         int
    batch_size:      int
    scramble_depth:  int
    lr:              float
    tau:             float
    tau_every:       int
    j_norm:          float
    weight_decay:    float
    max_update_loss: float

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
        env             = job.env,
        num_models      = job.num_models,
        batches         = job.batches,
        batch_size      = job.batch_size,
        scramble_depth  = job.scramble_depth,
        lr              = job.lr,
        tau             = job.tau,
        tau_every       = job.tau_every,
        j_norm          = job.j_norm,
        weight_decay    = job.weight_decay,
        max_update_loss = job.max_update_loss,
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
        model = Model(model_cfg).to(device)
        gen_model = Model(model_cfg).to(device)
        gen_model.eval()
        clone_model(model, gen_model)
        optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
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

    batches_per_gen = get_batches_per_gen(env, train_cfg.num_models * train_cfg.batch_size)
    log("Generating states every %i batches" % batches_per_gen)

    for i in range(train_cfg.batches):

        if i in eval_batches:
            TT.profile("Evaluate")
            log("Evaluating models")
            for model in models:
                model.eval()

            states, depths = gen_new_states(env, 10 ** 4, train_cfg.scramble_depth)
            states_d = states.to(device)
            states_oh = env.multiple_oh(states_d)
            with TT.profile("Value estimates"), torch.no_grad():
                preds = torch.zeros(len(states_d), device=device)
                for model in models:
                    preds += model(states_oh).squeeze()
                preds = train_cfg.j_norm * preds / len(models)

                for j in range(train_cfg.scramble_depth):
                    train_results.value_estimations[j].append(
                        preds[depths == j + 1].mean().item()
                    )

            for model in models:
                model.train()

            TT.end_profile()

        TT.profile("Batch")
        log("Batch %i / %i" % (i+1, train_cfg.batches))

        if i % batches_per_gen == 0:
            num_states = batches_per_gen * train_cfg.batch_size * train_cfg.num_models
            log(
                "Generating states:",
                "States:     %s" % thousands_seperators(num_states),
                "Neighbours: %s" % thousands_seperators(num_states * len(env.action_space)),
                "Total:      %s" % thousands_seperators(num_states * (1 + len(env.action_space))),
            )
            with TT.profile("Generate scrambled states"):
                all_states, _ = gen_new_states(
                    env,
                    num_states,
                    train_cfg.scramble_depth,
                )
            with TT.profile("Generate neighbour states"):
                all_neighbour_states = env.neighbours(all_states)

            all_states = all_states.view(
                batches_per_gen,
                train_cfg.num_models,
                train_cfg.batch_size,
                *env.get_solved().shape,
            )
            all_neighbour_states = all_neighbour_states.view(
                batches_per_gen,
                train_cfg.num_models,
                train_cfg.batch_size * len(env.action_space),
                *env.get_solved().shape,
            )
            log.debug("Size of states in bytes: %s" % thousands_seperators(tensor_size(all_states)))

        for j in range(train_cfg.num_models):
            log.debug("Training model %i / %i" % (j+1, train_cfg.num_models))

            model = models[j]
            gen_model = gen_models[j]
            optimizer = optimizers[j]
            scheduler = schedulers[j]

            with TT.profile("Transfer states to device"):
                states = all_states[i % batches_per_gen, j]
                states_d = states.to(device)

                neighbour_states = all_neighbour_states[i % batches_per_gen, j]
                neighbour_states_d = neighbour_states.to(device)

            log.debug("Forward passing states")
            TT.profile("Value estimates")
            with TT.profile("OH neighbour states"):
                neighbour_states_oh = env.multiple_oh(neighbour_states_d)
            assert neighbour_states_d.is_contiguous() and neighbour_states_oh.is_contiguous()

            with TT.profile("Forward pass"), torch.no_grad():
                J = gen_model(neighbour_states_oh).squeeze()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            with TT.profile("Set solved states to j = 0"):
                solved_states = env.multiple_is_solved(neighbour_states)
                J[solved_states] = 0
            J = J.view(len(states_d), len(env.action_space))

            with TT.profile("Calculate targets"):
                g = 1
                targets = torch.min(g + J, dim=1).values
            TT.end_profile()

            with TT.profile("OH states"):
                states_oh = env.multiple_oh(states_d)
            assert states_d.is_contiguous() and states_oh.is_contiguous()

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
                if i == 0:
                    log.debug(
                        "Shapes",
                        "states:              %s" % list(states_d.shape),
                        "states_oh:           %s" % list(states_oh.shape),
                        "neighbour_states:    %s" % list(neighbour_states_d.shape),
                        "neighbour_states_oh: %s" % list(neighbour_states_oh.shape),
                        "value_estimates:     %s" % list(J.shape),
                        "targets:             %s" % list(targets.shape),
                        sep="\n    ",
                    )
                    log.debug(
                        "Sizes in bytes",
                        "states:              %s" % thousands_seperators(tensor_size(states_d)),
                        "states_oh:           %s" % thousands_seperators(tensor_size(states_oh)),
                        "neighbour_states:    %s" % thousands_seperators(tensor_size(neighbour_states_d)),
                        "neighbour_states_oh: %s" % thousands_seperators(tensor_size(neighbour_states_oh)),
                        "value_estimates:     %s" % thousands_seperators(tensor_size(J)),
                        "targets:             %s" % thousands_seperators(tensor_size(targets)),
                        sep="\n    ",
                    )

            scheduler.step()

            if i % train_cfg.tau_every == 0 and train_results.losses[j][-1] < train_cfg.max_update_loss:
                log.debug("Updating generator network")
                with TT.profile("Update generator network"):
                    update_generator_network(train_cfg.tau, gen_models[j], models[j])

        log("Mean loss: %.4f" % (sum(ls[-1] for ls in train_results.losses) / train_cfg.num_models))

        TT.end_profile()

    log.section("Saving")
    with TT.profile("Save"):
        train_cfg.save(job.location)
        model_cfg.save(job.location)
        train_results.save(job.location)

        for i, model in enumerate(models):
            torch.save(model.state_dict(), f"{job.location}/model-{i}.pt")
