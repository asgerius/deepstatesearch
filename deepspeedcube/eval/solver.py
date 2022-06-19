from __future__ import annotations

import abc

import torch
from pelutils import TickTock

from deepspeedcube import device
from deepspeedcube.envs import Environment
from deepspeedcube.model import Model


class Solver(abc.ABC):

    def __init__(self, env: Environment, max_time: float | None):
        self.env = env
        self.max_time = max_time or 1e6
        self.tt = TickTock()

    @abc.abstractmethod
    def solve(self, state: torch.Tensor) -> torch.Tensor | None:
        """ Returns tensor of actions to solve the given state. """
        pass

class GreedyValueSolver(Solver):

    def __init__(self, env: Environment, max_time: float | None, models: list[Model]):
        super().__init__(env, max_time)
        self.models = models
        for model in self.models:
            model.eval()

    @torch.no_grad()
    def solve(self, state: torch.Tensor) -> tuple[torch.Tensor | None, float | None]:
        self.tt.tick()

        actions = list()
        while self.tt.tock() < self.max_time:
            neighbours = self.env.neighbours(state.unsqueeze(dim=0))
            solved = self.env.multiple_is_solved(neighbours)
            if torch.any(solved):
                actions.append(torch.where(solved)[0].item())
                return torch.tensor(actions), self.tt.tock()

            neighbours_oh = self.env.multiple_oh(neighbours)
            preds = torch.zeros(len(neighbours_oh), device=device)
            for model in self.models:
                preds += model(neighbours_oh).squeeze()

            action = preds.argmin().item()
            actions.append(action)

            state = self.env.move(action, state)

        return None, None

class AStar(Solver):

    _frontier_size = 10 ** 4

    def __init__(self, env: Environment, max_time: float | None, models: list[Model], l: float, N: int, d: int):
        super().__init__(env, max_time)
        self.models = models
        self.l = l
        self.N = N
        self.d = d

        for model in self.models:
            model.eval()

    @staticmethod
    def _expand_frontier(frontier: torch.Tensor):
        return torch.vstack((frontier, torch.empty_like(frontier)))

    @torch.no_grad()
    def solve(self, state: torch.Tensor) -> tuple[torch.Tensor | None, float | None]:
        self.tt.tick()

        self._frontier = torch.empty((self._frontier_size, len(state)), dtype=self.env.dtype, device=device)

        actions = list()
        while self.tt.tock() < self.max_time:
            pass

        return None, None
