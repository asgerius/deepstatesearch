from __future__ import annotations

import abc


class BaseEnvironment(abc.ABC):

    state_size: int


class _CubeEnvironment(BaseEnvironment):

    state_size = 480


_ENVS = {
    "cube": _CubeEnvironment,
}

def get_env(env: str) -> BaseEnvironment:
    return _ENVS[env]()
