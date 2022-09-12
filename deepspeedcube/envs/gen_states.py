from __future__ import annotations

from math import ceil

import psutil
import torch
from pelutils import TT, log, thousands_seperators

from deepspeedcube import tensor_size
from deepspeedcube.envs import Environment


def gen_new_states(env: Environment, num_states: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    K += 1  # Should be inclusive
    states_per_depth = ceil(num_states / K)

    with TT.profile("Create states"):
        states = env.get_multiple_solved(states_per_depth * K)
        scramble_depths = torch.zeros(len(states), dtype=torch.int32)

    with TT.profile("Scramble states"):
        for i in range(K):
            start = i * states_per_depth
            n = len(states) - start
            actions = torch.randint(
                0, len(env.action_space), (n,),
                dtype=torch.uint8,
            )
            env.multiple_moves(actions, states[start:], inplace=True)
            scramble_depths[start:] += 1

    with TT.profile("Shuffle states"):
        shuffle_index = torch.randperm(len(states))
        states[:] = states[shuffle_index]
        scramble_depths[:] = scramble_depths[shuffle_index]

    return states[:num_states], scramble_depths[:num_states]

def gen_eval_states(env: Environment, states_per_depth: int, depths: list[int]) -> torch.Tensor:
    total_states = states_per_depth * len(depths)

    with TT.profile("Create states"):
        states = env.get_multiple_solved(total_states)

    with TT.profile("Scramble states"):
        for i, (prev_depth, depth) in enumerate(zip([0, *depths[:-1]], depths)):
            start = i * states_per_depth
            n_states = total_states - start
            scrambles = depth - prev_depth
            for _ in range(scrambles):
                actions = torch.randint(
                    0, len(env.action_space), (n_states,),
                    dtype=torch.uint8,
                )
                env.multiple_moves(actions, states[start:], inplace=True)

    return states.view(len(depths), states_per_depth, *env.get_solved().shape)

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
