from __future__ import annotations

import abc
import contextlib
import ctypes

import torch
import torch.cuda.amp as amp
from pelutils import TickTock, TT

from deepstatesearch import device, LIBDSS, ptr, tensor_size
from deepstatesearch.envs import Environment
from deepstatesearch.model import Model


class Solver(abc.ABC):

    def __init__(self, env: Environment, max_time: float | None):
        self.env = env
        self.max_time = max_time or 1e6
        self.tt = TickTock()

    @abc.abstractmethod
    def solve(self, state: torch.Tensor) -> tuple[torch.Tensor | None, float, int]:
        """ Returns tensor of actions to solve the given state, the time spent is seconds, and the number of states seen. """
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

class GreedyValueSolver(Solver):

    def __init__(self, env: Environment, max_time: float | None, models: list[Model], fp16: bool):
        super().__init__(env, max_time)
        self.models = models
        for model in self.models:
            model.eval()
        self.fp16 = fp16

    @torch.no_grad()
    def solve(self, state: torch.Tensor) -> tuple[torch.Tensor | None, float, int]:
        self.tt.tick()

        state = state.clone()

        actions = list()
        states_seen = 1
        while self.tt.tock() < self.max_time:
            TT.profile("Iteration")

            _, neighbours = self.env.neighbours(state.unsqueeze(dim=0))
            # NB: This overestimates the number of states, as duplicates are not detected
            # This is somewhat corrected by removing one, which is the previous state
            states_seen += len(neighbours) - 1
            solved = self.env.multiple_is_solved(neighbours)
            if torch.any(solved):
                actions.append(torch.where(solved)[0].item())
                TT.end_profile()
                return torch.tensor(actions), self.tt.tock(), states_seen

            neighbours_d = neighbours.to(device)

            neighbours_oh = self.env.multiple_oh(neighbours_d)
            preds = torch.zeros(len(neighbours_oh), device=device)
            with amp.autocast() if self.fp16 else contextlib.ExitStack():
                for model in self.models:
                    preds += model(neighbours_oh).squeeze()

            action = preds.argmin().item()
            actions.append(action)

            state = neighbours[action]

            TT.end_profile()

        return None, self.tt.tock(), states_seen

    def __str__(self) -> str:
        return "Greedy Value"

class AStar(Solver):

    def __init__(self, env: Environment, max_time: float | None, models: list[Model], fp16: bool, l: float, N: int, d: int):
        super().__init__(env, max_time)
        self.models = models
        self.fp16 = fp16
        self.l = l  # lambda used to weigh moves spent and estimated cost-to-go
        self.N = N  # Number of states to expand each iteration
        self.d = d  # Parameter for the d-ary heap

        for model in self.models:
            model.eval()

    @torch.no_grad()
    def solve(self, state: torch.Tensor) -> tuple[torch.Tensor | None, float, int]:
        self.tt.tick()

        state = state.clone()

        if self.env.is_solved(state):
            return torch.tensor([], dtype=torch.long), self.tt.tock(), 1

        TT.profile("A*")

        with TT.profile("Allocate"):
            search_state_p = ctypes.c_void_p(LIBDSS.astar_init(
                ctypes.c_float(self.l),
                ctypes.c_size_t(self.d),
                ctypes.c_size_t(tensor_size(state)),
            ))
            frontier_p = ctypes.c_void_p(LIBDSS.astar_frontier_ptr(search_state_p))

        # Insert initial state into frontier and node map
        h = self.h(state.unsqueeze(0).to(device))[0].item()

        with TT.profile("Add initial state"):
            LIBDSS.astar_add_initial_state(h, ptr(state), search_state_p)

        solved = False

        while self.tt.tock() < self.max_time:

            TT.profile("Iteration")

            with TT.profile("Extract from frontier"):
                _, current_states = self.extract_min(frontier_p)
            with TT.profile("Get neighbours"):
                _, neighbour_states = self.env.neighbours(current_states)
            with TT.profile("To device"):
                neighbour_states_d = neighbour_states.to(device)

            # Make sure there is enough space in the heap, as astar_iteration
            # usually adds new states without allocating more memory
            while LIBDSS.heap_should_increase_alloc(frontier_p, ctypes.c_size_t(len(neighbour_states))):
                with TT.profile("Expand frontier"):
                    LIBDSS.heap_increase_alloc(frontier_p)

            with TT.profile("Check for solution"):
                any_solved = self.env.multiple_is_solved_d(neighbour_states_d).any()
            if any_solved:
                with TT.profile("Solve cleanup"):
                    longest_path = LIBDSS.astar_longest_path(search_state_p) + 1
                    actions = torch.empty(longest_path, dtype=torch.uint8)
                    solved_state_index = torch.where(self.env.multiple_is_solved(neighbour_states))[0][0].item()
                    actions[0] = solved_state_index % len(self.env.action_space)
                    final_state = current_states[solved_state_index // len(self.env.action_space)]
                    solved = True
                TT.end_profile()
                break

            h = self.h(neighbour_states_d)

            with TT.profile("Update search state"):
                LIBDSS.astar_iteration(
                    ctypes.c_size_t(len(current_states)),
                    ptr(current_states),
                    ctypes.c_size_t(len(neighbour_states)),
                    ptr(neighbour_states),
                    ptr(h),
                    ptr(self.env.action_space.repeat(len(current_states))),
                    search_state_p,
                )

            TT.end_profile()

        if solved:
            with TT.profile("Retrace path"):
                # Solved state has not been added, so add 1 to maximum solution length
                reverse_actions = self.env.reverse_moves(self.env.action_space)
                num_actions = LIBDSS.astar_retrace_path(
                    len(self.env.action_space),
                    ptr(actions),
                    ptr(reverse_actions),
                    ptr(final_state),
                    self.env.move_fn,
                    search_state_p,
                )
                actions = actions[:num_actions].flip(0)

        states_seen = LIBDSS.astar_num_states(search_state_p)

        with TT.profile("Free memory"):
            LIBDSS.astar_free(search_state_p)

        TT.end_profile()

        if solved:
            return actions, self.tt.tock(), states_seen

        return None, self.tt.tock(), states_seen

    def extract_min(self, frontier_p: ctypes.c_void_p) -> torch.Tensor:
        keys_arr = torch.empty((self.N, ), dtype=torch.float)
        data_arr = torch.empty((self.N, *self.env.state_shape), dtype=self.env.dtype)
        num_extracted = LIBDSS.heap_extract_min(frontier_p, self.N, ptr(keys_arr), ptr(data_arr))
        return keys_arr[:num_extracted], data_arr[:num_extracted]

    @torch.no_grad()
    def h(self, states_d: torch.Tensor) -> torch.Tensor:
        """ Calculates the average h produced by the models multiplied by lambda. """

        with TT.profile("One-hot"):
            states_oh = self.env.multiple_oh(states_d)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        preds = torch.zeros(len(states_d), dtype=torch.float, device=device)
        with TT.profile("Estimate cost-to-go"), \
            amp.autocast() if self.fp16 and torch.cuda.is_available() else contextlib.ExitStack():
            for model in self.models:
                preds += model(states_oh).squeeze()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return preds.cpu() / len(self.models)

    def __str__(self) -> str:
        return f"$A^*(\lambda={self.l}, N={self.N}, d={self.d})$"
