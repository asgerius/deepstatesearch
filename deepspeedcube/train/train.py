from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from pelutils import log, thousands_seperators, TT
from pelutils.datastorage import DataStorage
from pelutils.parser import JobDescription

from deepspeedcube import LIBDSC, device, ptr, tensor_size
from deepspeedcube.model import Model, ModelConfig
from deepspeedcube.envs import NULL_ACTION, get_env
from deepspeedcube.envs.gen_states import gen_new_states, get_batches_per_gen
from deepspeedcube.model.generator_network import clone_model, update_generator_network


@dataclass
class TrainConfig(DataStorage, json_name="train_config.json", indent=4):
    env:                str
    num_models:         int
    batches:            int
    batch_size:         int
    K:                  int
    lr:                 float
    tau:                float
    tau_every:          int
    weight_decay:       float
    epsilon:            float
    known_states_depth: int
    fp16:               bool

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
        env                 = job.env,
        num_models          = job.num_models,
        batches             = job.batches,
        batch_size          = job.batch_size,
        K                   = job.k,
        lr                  = job.lr,
        tau                 = job.tau,
        tau_every           = job.tau_every,
        weight_decay        = job.weight_decay,
        epsilon             = job.epsilon,
        known_states_depth  = job.known_states_depth,
        fp16                = job.fp16,
    )
    train_cfg.fp16 = train_cfg.fp16 and torch.cuda.is_available()
    log("Got training config", train_cfg)

    log("Setting up environment '%s'" % train_cfg.env)
    env = get_env(train_cfg.env)

    eval_batches = np.arange(0, train_cfg.batches, 1000, dtype=int).tolist()
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
        j_norm              = job.j_norm,
    )
    log("Got model config", model_cfg)

    criterion = nn.MSELoss()
    models: list[Model] = list()
    gen_models: list[Model] = list()
    optimizers = list()
    schedulers = list()
    scalers = list()
    for _ in range(train_cfg.num_models):
        TT.profile("Build model")
        model = Model(model_cfg).to(device)
        gen_model = Model(model_cfg).to(device)
        gen_model.eval()
        clone_model(model, gen_model)
        optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg.batches)
        scaler = amp.grad_scaler.GradScaler() if train_cfg.fp16 else None

        models.append(model)
        gen_models.append(gen_model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        scalers.append(scaler)
        TT.end_profile()
    log(
        "Built %i models" % train_cfg.num_models,
        "Parameters per model: %s" % thousands_seperators(models[0].numel()),
        "Total parameters:     %s" % thousands_seperators(sum(m.numel() for m in models)),
    )
    log(models[0], "Size in bytes: %s" % thousands_seperators(4*models[0].numel()))

    log.section("Starting training")
    train_results = TrainResults(eval_idx=eval_batches)
    train_results.value_estimations.extend(list() for _ in range(train_cfg.K))
    train_results.losses.extend(list() for _ in range(train_cfg.num_models))

    batches_per_gen = get_batches_per_gen(env, train_cfg.num_models * train_cfg.batch_size)
    log("Generating states every %i batches" % batches_per_gen)

    num_states_update = train_cfg.batches * train_cfg.batch_size
    num_states_total = num_states_update * (1 + len(env.action_space))
    log("During training, each model will perform updates from %s cost-to-go estimates and see %s states in total" % (
        thousands_seperators(num_states_update), thousands_seperators(num_states_total)
    ))

    if train_cfg.known_states_depth:
        log("Generating easy, known states up to depth %i" % train_cfg.known_states_depth)
        with TT.profile("Generate known states"):
            num_states = (len(env.action_space) ** np.arange(train_cfg.known_states_depth + 1)).sum()
            known_states = torch.empty((num_states, *env.state_shape), dtype=env.dtype)
            known_states[0] = env.get_solved()
            start = 1
            current_num_states = 1
            for i in range(1, train_cfg.known_states_depth+1):
                known_states[start:start + len(env.action_space) * current_num_states] \
                    = env.neighbours(known_states[start - current_num_states:start])[1]
                current_num_states *= len(env.action_space)
                start += current_num_states

        log("Inserting known states into map")
        with TT.profile("Insert known states"):
            known_states_map_p = ctypes.c_void_p(LIBDSC.values_node_map_from_states(
                len(known_states), env.state_size,
                ptr(known_states), len(env.action_space),
            ))

        log(
            "Generated %s easy, known states" % thousands_seperators(num_states),
            "Size of state array in bytes: %s" % thousands_seperators(tensor_size(known_states))
        )

    for i in range(train_cfg.batches):

        if i in eval_batches:
            TT.profile("Evaluate")
            log("Evaluating value estimates")
            for model in models:
                model.eval()

            states, depths = gen_new_states(env, min(train_cfg.K * 10 ** 3, 5 * 10 ** 4), train_cfg.K)
            states_d = states.to(device)
            states_oh = env.multiple_oh(states_d)
            with TT.profile("Value estimates"), torch.no_grad(), amp.autocast() if train_cfg.fp16 else contextlib.ExitStack():
                preds = torch.zeros(len(states_d), device=device)
                for model in models:
                    preds += model(states_oh).squeeze()
                preds = preds / len(models)

                for j in range(train_cfg.K):
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
                    train_cfg.K,
                )
            with TT.profile("Generate neighbour states"):
                all_to_neighbour_actions, all_neighbour_states = env.neighbours(all_states)

            all_states = all_states.view(
                batches_per_gen,
                train_cfg.num_models,
                train_cfg.batch_size,
                *env.get_solved().shape,
            )
            all_to_neighbour_actions = all_to_neighbour_actions.view(
                batches_per_gen,
                train_cfg.num_models,
                train_cfg.batch_size * len(env.action_space),
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
                # print(states.device)
                # print("0x%x" % ptr(states).value)
                states_d = states.to(device)
                # print(states.device)

                to_neighbour_actions_d = all_to_neighbour_actions[i % batches_per_gen, j].to(device)

                neighbour_states = all_neighbour_states[i % batches_per_gen, j]
                neighbour_states_d = neighbour_states.to(device)

            log.debug("Forward passing states")
            TT.profile("Value estimates")
            with TT.profile("OH neighbour states"):
                neighbour_states_oh = env.multiple_oh(neighbour_states_d)
            assert neighbour_states_d.is_contiguous() and neighbour_states_oh.is_contiguous()

            with TT.profile("Forward pass"), torch.no_grad(), amp.autocast() if train_cfg.fp16 else contextlib.ExitStack():
                J = gen_model(neighbour_states_oh).squeeze()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            if not train_cfg.known_states_depth:
                with TT.profile("Set solved states to j = 0"):
                    solved_states = env.multiple_is_solved_d(neighbour_states_d)
                    J[solved_states] = 0
            J = J.view(len(states_d), len(env.action_space))

            with TT.profile("Calculate targets"):
                # Set g to 1 for valid moves and effectively inf for invalid moves.
                # Effectively the same as not including the moves, but allows
                # for better and easier vectorization.
                # Note: Do not use torch.inf, as 0 * torch.inf is nan.
                g = (to_neighbour_actions_d == NULL_ACTION) \
                    .to(torch.float) \
                    .view(len(states_d), len(env.action_space))
                g = g * 1e10 + 1
                targets = torch.min(g + J, dim=1).values
            TT.end_profile()

            if train_cfg.known_states_depth:
                with TT.profile("Set known state values"):
                    targets_cpu = targets.cpu()
                    LIBDSC.values_set(len(targets_cpu), env.state_size, ptr(states), ptr(targets_cpu), known_states_map_p)
                    targets = targets_cpu.to(device)

            with TT.profile("OH states"):
                states_oh = env.multiple_oh(states_d)
            assert states_d.is_contiguous() and states_oh.is_contiguous()

            with TT.profile("Train model"), amp.autocast() if train_cfg.fp16 else contextlib.ExitStack():
                preds = model(states_oh).squeeze()
                loss = criterion(preds, targets)
                if train_cfg.fp16:
                    scalers[j].scale(loss).backward()
                    scalers[j].step(optimizer)
                    scalers[j].update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
                log.debug("Loss: %.4f" % loss.item())
                train_results.losses[j].append(loss.item())
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

            if i % train_cfg.tau_every == 0 and train_results.losses[j][-1] < train_cfg.epsilon:
                log("Updating generator network")
                with TT.profile("Update generator network"):
                    update_generator_network(train_cfg.tau, gen_models[j], models[j])

        log("Mean loss: %.4f" % (sum(ls[-1] for ls in train_results.losses) / train_cfg.num_models))

        TT.end_profile()

    if train_cfg.known_states_depth:
        LIBDSC.values_free(known_states_map_p)

    log.section("Saving")
    with TT.profile("Save"):
        train_cfg.save(job.location)
        model_cfg.save(job.location)
        train_results.save(job.location)

        for i, model in enumerate(models):
            torch.save(model.state_dict(), f"{job.location}/model-{i}.pt")
