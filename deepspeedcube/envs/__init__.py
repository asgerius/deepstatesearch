from __future__ import annotations

import abc
import ctypes
from typing import Type

import numpy as np
import torch
import torch.nn.functional as F

from deepspeedcube import device, ptr, tensor_size, LIBDSC


NULL_ACTION = 255

class Environment(abc.ABC):

    dtype: type
    action_space: torch.Tensor
    state_shape: torch.Size
    state_oh_size: int
    _solved_state: torch.Tensor
    _solved_state_d: torch.Tensor
    state_size: int
    move_fn: ctypes._NamedFuncPointer

    @classmethod
    def get_solved(cls) -> torch.Tensor:
        return cls._solved_state.clone()

    @classmethod
    def get_multiple_solved(cls, n: int) -> torch.Tensor:
        return cls._solved_state.repeat((n, *[1]*len(cls._solved_state.shape)))

    @classmethod
    def is_solved(cls, state: torch.Tensor) -> bool:
        return torch.all(state == cls._solved_state)

    @classmethod
    def multiple_is_solved(cls, states: torch.Tensor) -> torch.BoolTensor:
        return (states == cls._solved_state).all(dim=1)

    @classmethod
    def multiple_is_solved_d(cls, states_d: torch.Tensor) -> torch.BoolTensor:
        return (states_d == cls._solved_state_d).all(dim=1)

    @classmethod
    def move(cls, action: int, state: torch.Tensor) -> torch.Tensor:
        new_state = state.clone()

        cls.move_fn(ptr(new_state), ctypes.c_char(action))

        return new_state

    @abc.abstractclassmethod
    def multiple_moves(cls, actions: torch.Tensor, states: torch.Tensor, inplace=False) -> torch.Tensor:
        pass

    @classmethod
    def neighbours(cls, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        neighbour_states = states.repeat_interleave(len(cls.action_space), dim=0)
        actions = cls.action_space.repeat(len(states))
        cls.multiple_moves(actions, neighbour_states, inplace=True)
        return actions, neighbour_states

    @classmethod
    def oh(cls, state: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            state.long(),
            num_classes=cls.state_oh_size // len(cls._solved_state),
        ).to(torch.float32).view(1, -1)

    @classmethod
    def multiple_oh(cls, states: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            states.long(),
            num_classes=cls.state_oh_size // len(cls._solved_state),
        ).to(torch.float32).view(len(states), -1)

    @abc.abstractclassmethod
    def reverse_move(cls, action: int) -> int:
        pass

    @abc.abstractclassmethod
    def reverse_moves(cls, actions: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractclassmethod
    def string(cls, state: torch.Tensor) -> str:
        pass

class _CubeEnvironment(Environment):

    dtype = torch.uint8
    action_space = torch.arange(12, dtype=torch.uint8)
    state_shape = torch.Size([20])
    state_oh_size = 480
    _solved_state = torch.tensor(
        [0, 3, 6, 9, 12, 15, 18, 21,
         0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        dtype=dtype,
    )
    _solved_state_d = _solved_state.to(device)
    state_size = tensor_size(_solved_state)
    move_fn = LIBDSC.cube_act

    # If the six sides are represented by an array, the order should be F, B, T, D, L, R
    F, B, T, D, L, R = 0, 1, 2, 3, 4, 5
    corner_633map = (
        ((F, 0, 0), (L, 0, 2), (T, 2, 0)),
        ((F, 2, 0), (D, 0, 0), (L, 2, 2)),
        ((F, 2, 2), (R, 2, 0), (D, 0, 2)),
        ((F, 0, 2), (T, 2, 2), (R, 0, 0)),
        ((B, 0, 2), (T, 0, 0), (L, 0, 0)),
        ((B, 2, 2), (L, 2, 0), (D, 2, 0)),
        ((B, 2, 0), (D, 2, 2), (R, 2, 2)),
        ((B, 0, 0), (R, 0, 2), (T, 0, 2)),
    )
    side_633map = (
        ((F, 0, 1), (T, 2, 1)),
        ((F, 1, 0), (L, 1, 2)),
        ((F, 2, 1), (D, 0, 1)),
        ((F, 1, 2), (R, 1, 0)),
        ((T, 1, 0), (L, 0, 1)),
        ((D, 1, 0), (L, 2, 1)),
        ((D, 1, 2), (R, 2, 1)),
        ((T, 1, 2), (R, 0, 1)),
        ((B, 0, 1), (T, 0, 1)),
        ((B, 1, 2), (L, 1, 0)),
        ((B, 2, 1), (D, 2, 1)),
        ((B, 1, 0), (R, 1, 2)),
    )

    @classmethod
    def multiple_moves(cls, actions: torch.Tensor, states: torch.Tensor, inplace=False) -> torch.Tensor:
        new_states = states if inplace else states.clone()

        LIBDSC.cube_multi_act(
            ptr(new_states),
            ptr(actions),
            len(actions),
        )

        return new_states

    @classmethod
    def reverse_move(cls, action: int) -> int:
        return action + 6 if action < 6 else action - 6

    @classmethod
    def reverse_moves(cls, actions: torch.Tensor) -> torch.Tensor:
        pos_dir = actions >= 6
        actions = actions + 6
        actions[pos_dir] -= 12
        return actions

    @classmethod
    def _as633(cls, state: torch.Tensor) -> np.ndarray:

        state = state.numpy()
        state633 = (np.ones((3, 3, 6)) * np.arange(6)).transpose(2, 1, 0).astype(int)
        for i in range(8):
            # Inserts values for corner i in position pos
            pos = state[i] // 3
            orientation = state[i] % 3
            # For these corners, "right turn" order is 0 2 1 instead of 0 1 2, so orientation is messed up without this fix
            if pos in [0, 2, 5, 7]:
                orientation *= -1
            values = np.roll([x[0] for x in cls.corner_633map[i]], orientation)
            state633[cls.corner_633map[pos][0]] = values[0]
            state633[cls.corner_633map[pos][1]] = values[1]
            state633[cls.corner_633map[pos][2]] = values[2]

        for i in range(12):
            # Inserts values for side i in position pos
            pos = state[i + 8] // 2
            orientation = state[i + 8] % 2
            values = np.roll([x[0] for x in cls.side_633map[i]], orientation)
            state633[cls.side_633map[pos][0]] = values[0]
            state633[cls.side_633map[pos][1]] = values[1]

        return state633

    @classmethod
    def _stringify_cube(cls, state633: np.ndarray) -> str:
        stringarr = np.empty((9, 12), dtype=str)
        stringarr[...] = " "
        simple = np.array([
            [-1, cls.T, -1, -1],
            [cls.L, cls.F, cls.R, cls.B],
            [-1, cls.D, -1, -1],
        ], dtype=int)
        for i in range(6):
            pos = tuple(int(x) for x in np.where(simple == i))
            stringarr[pos[0] * 3: pos[0] * 3 + 3, pos[1] * 3: pos[1] * 3 + 3] = state633[i].astype(str)
        string = "\n".join([" ".join(list(y)) for y in stringarr])
        return string

    @classmethod
    def string(cls, state: torch.Tensor) -> str:
        return cls._stringify_cube(cls._as633(state.cpu()))

class _SlidingPuzzle(Environment):

    dtype = torch.int8
    action_space = torch.arange(4, dtype=torch.uint8)

    move_fn = LIBDSC.sliding_act

    def __init_subclass__(cls, size: int):
        cls.size = ctypes.c_int(size)
        cls.state_shape = torch.Size([3 + size ** 2])
        cls.state_oh_size = size ** 4
        cls._solved_state = torch.concat((
            torch.tensor([0, 0, size]), torch.arange(size ** 2)
        )).to(_SlidingPuzzle.dtype)
        cls._solved_state_d = cls._solved_state.to(device)
        cls.state_size = tensor_size(cls._solved_state)

    @classmethod
    def multiple_moves(cls, actions: torch.Tensor, states: torch.Tensor, inplace=False) -> torch.Tensor:
        states = states if inplace else states.clone()

        LIBDSC.sliding_multi_act(
            ptr(states),
            ptr(actions),
            ctypes.c_size_t(len(actions)),
        )

        return states

    @classmethod
    def neighbours(cls, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions, neighbour_states = super().neighbours(states)
        LIBDSC.sliding_neighbours_set_null_actions(
            ptr(states),
            ptr(neighbour_states),
            ptr(actions),
            ctypes.c_size_t(len(states)),
        )
        return actions, neighbour_states

    @classmethod
    def oh(cls, state: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            state[3:].long(),
            num_classes=cls.state_oh_size // len(cls._solved_state[3:]),
        ).astype(torch.float32).view(1, -1)

    @classmethod
    def multiple_oh(cls, states: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            states[:, 3:].long(),
            num_classes=cls.state_oh_size // len(cls._solved_state[3:]),
        ).to(torch.float32).view(len(states), -1)

    @classmethod
    def reverse_move(cls, action: int) -> int:
        return (action + 2) % 4

    @classmethod
    def reverse_moves(cls, actions: torch.Tensor) -> torch.Tensor:
        return (actions + 2) % 4

    @classmethod
    def string(cls, state: torch.Tensor) -> str:
        size = cls.size.value
        readable_state = np.empty((size, size), dtype=int)
        for i in range(size ** 2):
            row = i // size
            col = i % size
            readable_state[row, col] = state[i+3]
        return str(readable_state)

class _SlidingPuzzle15(_SlidingPuzzle, size=4):
    pass

class _SlidingPuzzle24(_SlidingPuzzle, size=5):
    pass

class _SlidingPuzzle35(_SlidingPuzzle, size=6):
    pass

class _SlidingPuzzle48(_SlidingPuzzle, size=7):
    pass

class _SlidingPuzzle63(_SlidingPuzzle, size=8):
    pass


_ENVS = {
    "cube": _CubeEnvironment,
    "sp15": _SlidingPuzzle15,
    "sp24": _SlidingPuzzle24,
    "sp35": _SlidingPuzzle35,
    "sp48": _SlidingPuzzle48,
    "sp63": _SlidingPuzzle63,
}

def get_env(env: str) -> Type[Environment]:
    return _ENVS[env]
