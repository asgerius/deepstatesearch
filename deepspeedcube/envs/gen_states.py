from __future__ import annotations

from math import ceil

import torch
from pelutils import TT

from deepspeedcube import device
from deepspeedcube.envs import BaseEnvironment


def gen_new_states(env: BaseEnvironment, num_states: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    states_per_depth = ceil(num_states / K)

    with TT.profile("Generate states"):
        states = env.get_multiple_solved(states_per_depth*K)
        scramble_depths = torch.zeros(len(states), dtype=torch.int16, device=device)

    with TT.profile("Scramble states"):
        for i in range(K):
            start = i * states_per_depth
            n = len(states) - start
            actions = torch.randint(
                0, len(env.action_space), (n,),
                dtype=torch.uint8,
                device=device,
            )
            env.multiple_moves(actions, states[start:], inplace=True)
            scramble_depths[start:] += 1

    with TT.profile("Shuffle states"):
        shuffle_index = torch.randperm(len(states), device=device)
        states[:] = states[shuffle_index]
        scramble_depths[:] = scramble_depths[shuffle_index]

    return states, scramble_depths

