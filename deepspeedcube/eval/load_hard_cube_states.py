from __future__ import annotations

from typing import Type

import torch

from deepspeedcube.envs import get_env


_env = get_env("cube")
_action_map = "F", "B", "U", "D", "L", "R"

def load_cube_eval_states(path: str) -> tuple[torch.Tensor, list[int]]:

    states = list()
    depths = list()

    with open(path) as f:
        for line in f:
            scramble = line.strip()
            if not scramble:
                continue

            assert len(scramble) % 2 == 0
            depth = len(scramble) // 2
            state = _env.get_solved()
            for i in range(depth):
                act = _action_map.index(scramble[i*2])
                num = int(scramble[i*2+1])
                for _ in range(num):
                    state = _env.move(act, state)

            states.append(state)
            depths.append(depth)

    return torch.vstack(states), depths


def load_hard_and_intermediate_states(path: str) -> torch.Tensor:
    """ Loads states given by the scrambles in path up to distance 24
    as well as intermediate states. """

    with open(path) as f:
        lines = [x for line in f if (x := line.strip())]
        lines = [l for l in lines if len(l) == 2 * 24]

    states = torch.empty((25, len(lines), *_env.state_shape), dtype=_env.dtype)
    states[0] = _env.get_multiple_solved(len(lines))

    for i, scramble in enumerate(lines):
        for j in range(1, 25):

            act = _action_map.index(scramble[(j-1)*2])
            num = int(scramble[(j-1)*2+1])
            states[j, i] = states[j-1, i]
            for _ in range(num):
                states[j, i] = _env.move(act, states[j, i])

    return states
