from __future__ import annotations

import abc
import ctypes

import numpy as np
from pelutils import c_ptr


_CUBELIB = ctypes.cdll.LoadLibrary("lib/cube.so")

class BaseEnvironment(abc.ABC):

    state_oh_size: int

    @abc.abstractclassmethod
    def move(cls, action: int, state: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def multiple_moves(cls, actions: np.ndarray, states: np.ndarray) -> np.ndarray:
        pass

class _CubeEnvironment(BaseEnvironment):

    state_oh_size = 480

    @classmethod
    def move(cls, action: int, state: np.ndarray) -> np.ndarray:
        new_state = state.copy()
        _CUBELIB.multi_act(
            c_ptr(new_state),
            c_ptr(np.array([action], dtype=np.uint8)),
            1,
        )
        return new_state

    @classmethod
    def multiple_moves(cls, actions: np.ndarray, states: np.ndarray) -> np.ndarray:
        new_states = states.copy()
        _CUBELIB.multi_act(
            c_ptr(new_states),
            c_ptr(actions),
            len(actions),
        )
        return new_states

_ENVS = {
    "cube": _CubeEnvironment,
}

def get_env(env: str) -> BaseEnvironment:
    return _ENVS[env]()
