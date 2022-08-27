from __future__ import annotations

from dataclasses import dataclass

import torch
from pelutils import DataStorage, log, TT, thousands_seperators
from pelutils.parser import JobDescription

from deepspeedcube import device
from deepspeedcube.envs import get_env
from deepspeedcube.envs.gen_states import gen_eval_states
from deepspeedcube.eval.solver import AStar, GreedyValueSolver
from deepspeedcube.model import ModelConfig, Model
from deepspeedcube.train.train import TrainConfig


@dataclass
class EvalConfig(DataStorage, json_name="eval_cfg.json", indent=4):
    solver:           str
    depths:           list[int]
    states_per_depth: int
    max_time:         float
    astar_lambda:     float
    astar_n:          int
    astar_d:          int
    validate:         bool
    solver_name:      str

@dataclass
class EvalResults(DataStorage, json_name="eval_results.json", indent=4):
    num_solved:    list[int]
    solved:        list[list[bool]]
    solve_times:   list[list[float]]
    states_seen:   list[list[int]]
    solve_lengths: list[list[int | None]]

@torch.no_grad()
def eval(job: JobDescription):
    log.section("Loading configurations")
    eval_cfg = EvalConfig(
        solver           = job.solver,
        depths           = list(range(job.max_depth+1)),
        states_per_depth = job.states_per_depth,
        max_time         = job.max_time,
        astar_lambda     = job.astar_lambda,
        astar_n          = job.astar_n,
        astar_d          = job.astar_d,
        validate         = job.validate,
        solver_name      = ""  # Set later
    )
    log("Got eval config", eval_cfg)

    train_cfg = TrainConfig.load(f"{job.location}/..")
    log("Got train config", train_cfg)

    model_cfg = ModelConfig.load(f"{job.location}/..")
    log("Got model config", model_cfg)

    log("Loading environment %s" % train_cfg.env)
    env = get_env(train_cfg.env)

    log.section("Loading %i models" % train_cfg.num_models)
    models = list()
    for i in range(train_cfg.num_models):
        TT.profile("Load model")
        log(f"Loading model {i+1} / {train_cfg.num_models}")
        model = Model(model_cfg).to(device)
        model.load_state_dict(torch.load(
            f"{job.location}/../model-{i}.pt",
            map_location=device,
        ))
        model.eval()
        models.append(model)
        TT.end_profile()

        if i == 0:
            log.debug(
                "Parameters per model: %s" % thousands_seperators(model.numel()),
                "Total parameters:     %s" % thousands_seperators(model.numel() * train_cfg.num_models),
            )
            log.debug(model)

    log.section("Preparing solver")
    if eval_cfg.solver == "GreedyValueSolver":
        solver = GreedyValueSolver(env, eval_cfg.max_time, models)
    elif eval_cfg.solver == "AStar":
        solver = AStar(env, eval_cfg.max_time, models,
            eval_cfg.astar_lambda, eval_cfg.astar_n, eval_cfg.astar_d,
        )
    else:
        raise ValueError("Unknown solver '%s'" % eval_cfg.solver)
    eval_cfg.solver_name = str(solver)

    results = EvalResults(
        num_solved    = list(),
        solved        = list(),
        solve_times   = list(),
        states_seen   = list(),
        solve_lengths = list(),
    )

    log.section("Evaluating")
    states = gen_eval_states(env, eval_cfg.states_per_depth, eval_cfg.depths)
    for i, depth in enumerate(eval_cfg.depths):
        log("Evaluating at depth %i" % depth)
        results.num_solved.append(0)
        results.solved.append(list())
        results.solve_times.append(list())
        results.states_seen.append(list())
        results.solve_lengths.append(list())

        TT.profile("Evaluate at depth %i" % depth)

        for state in states[i]:
            actions, time, states_seen = solver.solve(state)
            did_solve = actions is not None
            results.num_solved[-1] += did_solve
            results.solved[-1].append(did_solve)
            results.solve_times[-1].append(time)
            results.states_seen[-1].append(states_seen)
            results.solve_lengths[-1].append(len(actions) if did_solve else -1)

            if did_solve and eval_cfg.validate:
                TT.profile("Validate")
                new_state = state
                for action in actions:
                    new_state = env.move(action, new_state)
                if not env.is_solved(new_state):
                    log.error(
                        "State %s was not solved" % state,
                        "Actions:     %s" % actions,
                        "Final state: %s" % new_state,
                    )
                    raise RuntimeError("Failed to solve state")
                TT.end_profile()

        log.debug("Solved %i / %i = %.0f %%" % (
            results.num_solved[-1],
            eval_cfg.states_per_depth,
            100 * results.num_solved[-1] / eval_cfg.states_per_depth,
        ))
        solved_states = torch.BoolTensor(results.solved[-1])
        if solved_states.any():
            # Log statistics for solved states
            solve_times = torch.FloatTensor(results.solve_times[-1])[solved_states]
            states_seen = torch.FloatTensor(results.states_seen[-1])[solved_states]
            solve_lengths = torch.FloatTensor(results.solve_lengths[-1])[solved_states]
            log.debug(
                "For the solved states:",
                "Avg. solution time:    %.2f ms" % (1000 * solve_times.mean()),
                "Avg. states seen:      %s" % thousands_seperators(round(states_seen.mean().item())),
                "Avg. nodes per second: %s" % thousands_seperators(round(states_seen.mean().item() / solve_times.mean().item())),
                "Avg. solution length:  %.2f" % solve_lengths.mean(),
                sep="\n    ",
            )

        TT.end_profile()

    log.section("Saving")
    eval_cfg.save(job.location)
    results.save(job.location)
