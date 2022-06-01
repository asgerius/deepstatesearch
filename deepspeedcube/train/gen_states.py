from __future__ import annotations

from math import ceil

import torch
from pelutils import TT

from deepspeedcube import device
from deepspeedcube.envs import BaseEnvironment


def gen_new_states(env: BaseEnvironment, num_states: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    states_per_depth = ceil(num_states / K)

    with TT.profile("Generate states"):
        states = env.get_multiple_solved(states_per_depth*K).to(device)
        scramble_depths = torch.ones(len(states), dtype=int)

    with TT.profile("Scramble states"):
        for i in range(K):
            start = i * states_per_depth
            n = len(states) - start
            actions = torch.randint(0, len(env.action_space), (n,), dtype=torch.uint8)
            env.multiple_moves(actions, states[start:], inplace=True)
            scramble_depths[start:] += 1

    with TT.profile("Shuffle states"):
        shuffle_index = torch.randperm(len(states))
        states = states[shuffle_index][:num_states]
        scramble_depths = scramble_depths[shuffle_index][:num_states]

    return states, scramble_depths

