from __future__ import annotations

from typing import Type

import torch

from deepspeedcube.envs import get_env


def load_cube_eval_states(path: str) -> tuple[torch.Tensor, list[int]]:

    env = get_env("cube")
    action_map = "F", "B", "U", "D", "L", "R"

    states = list()
    depths = list()

    with open(path) as f:
        for line in f:
            scramble = line.strip()
            if not scramble:
                continue

            assert len(scramble) % 2 == 0
            depth = len(scramble) // 2
            state = env.get_solved()
            for i in range(depth):
                act = action_map.index(scramble[i*2])
                num = int(scramble[i*2+1])
                for _ in range(num):
                    state = env.move(act, state)

            states.append(state)
            depths.append(depth)

    return torch.vstack(states), depths
