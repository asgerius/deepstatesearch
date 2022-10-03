from __future__ import annotations

import random
from math import ceil

import psutil
import torch
from pelutils import TT, log, thousands_seperators

from deepspeedcube import device, tensor_size
from deepspeedcube.envs import Environment


def gen_new_states(env: Environment, num_states: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    if num_states % K:
        num_states = K * (num_states // K)
    states_per_depth = num_states // K

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

    with TT.profile("Reorganize states"):
        with TT.profile("Index"):
            index = (torch.arange(num_states, device=device) * states_per_depth) % num_states \
                + torch.repeat_interleave(torch.arange(states_per_depth, device=device), num_states // states_per_depth)
        with TT.profile("Reorganize"):
            states = states[index].contiguous()
            scramble_depths = scramble_depths[index].contiguous()

    return states, scramble_depths

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
    # Only generate up to 20 GB of states, though no more than half the available memory
    available_memory = 20 * 10 ** 9
    if available_memory > (max_mem := psutil.virtual_memory().total / 2):
        available_memory = max_mem

    # Calculate memory requirements for scrambling
    state_memory           = 2 * tensor_size(env.get_solved())
    scramble_depths_memory = 4  # int32
    actions_memory         = 1  # uint8
    shuffle_index_memory   = 8  # int64
    scramble_memory        = state_memory + scramble_depths_memory\
        + actions_memory + shuffle_index_memory

    # Calculate memory requirements for getting neighbour states
    state_memory     = tensor_size(env.get_solved()) * len(env.action_space)
    actions_memory   = tensor_size(env.action_space)
    neighbour_memory = state_memory / 2 + actions_memory

    total_batch_memory = batch_size * (scramble_memory + neighbour_memory)
    log.debug(
        "Memory requirements for generating states for a batch of size %i:" % batch_size,
        thousands_seperators(total_batch_memory // 2 ** 20) + " MB",
    )

    num_batches = available_memory // total_batch_memory

    return num_batches
