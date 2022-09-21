from __future__ import annotations

import random
from math import ceil

import psutil
import torch
from pelutils import TT, log, thousands_seperators

from deepspeedcube import device, tensor_size
from deepspeedcube.envs import Environment


def gen_new_states(env: Environment, num_states: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    states_per_depth = ceil(num_states / K)

    with TT.profile("Create states"):
        states = env.get_multiple_solved(states_per_depth * K)
        scramble_depths = torch.zeros(len(states), dtype=torch.int32)

    with TT.profile("Scramble states"):
        for i in range(K):
            start = i * states_per_depth
            n = len(states) - start
            with TT.profile("Generate actions"):
                actions = torch.randint(
                    0, len(env.action_space), (n,),
                    dtype=torch.uint8,
                    device=device,
                ).cpu()
            with TT.profile("Perform actions"):
                env.multiple_moves(actions, states[start:], inplace=True)
            scramble_depths[start:] += 1

    with TT.profile("Shuffle states"):
        shuffle_index = torch.randperm(len(states), device=device).cpu()
        states[:] = states[shuffle_index]
        scramble_depths[:] = scramble_depths[shuffle_index]

    return states[:num_states], scramble_depths[:num_states]

def gen_eval_states(env: Environment, num_states: int, min_scrambles: int, max_scrambles: int) -> tuple[torch.Tensor, list[int]]:
    depths = [random.randint(min_scrambles, max_scrambles) for _ in range(num_states)]

    with TT.profile("Create states"):
        states = env.get_multiple_solved(num_states)

    with TT.profile("Scramble states"):
        for i, d in enumerate(depths):
            for _ in range(d):
                states[i] = env.move(random.randint(0, len(env.action_space) - 1), states[i])

    return states, depths

def get_batches_per_gen(env: Environment, batch_size: int) -> int:
    max_gen_states = 100 * 10 ** 6

    # Calculate memory requirements for scrambling
    state_memory           = tensor_size(env.get_solved())
    scramble_depths_memory = 4  # int32
    actions_memory         = 1  # uint8
    shuffle_index_memory   = 8  # int64
    scramble_memory        = state_memory + scramble_depths_memory\
        + actions_memory + shuffle_index_memory

    # Calculate memory requirements for getting neighbour states
    state_memory     = tensor_size(env.get_solved()) * len(env.action_space)
    actions_memory   = tensor_size(env.action_space)
    neighbour_memory = state_memory + actions_memory

    total_batch_memory = batch_size * (scramble_memory + neighbour_memory)
    log.debug(
        "Memory requirements for generating states for a batch of size %i:" % batch_size,
        thousands_seperators(total_batch_memory // 2 ** 20) + " MB",
    )

    avail_mem = psutil.virtual_memory().total
    max_memory_frac = 0.5
    avail_mem *= max_memory_frac

    num_batches = avail_mem // total_batch_memory
    if num_batches * batch_size * (1 + len(env.action_space)) > max_gen_states:
        num_batches = max_gen_states // (batch_size * (1 + len(env.action_space)))

    return num_batches
