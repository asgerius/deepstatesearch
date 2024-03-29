from __future__ import annotations

import math
import os
from dataclasses import dataclass

import psutil
import torch
from pelutils import DataStorage, log, TT, thousands_seperators
from pelutils.parser import JobDescription
from pelutils.ds.stats import z

from deepstatesearch import device
from deepstatesearch.envs import get_env
from deepstatesearch.envs.gen_states import gen_eval_states
from deepstatesearch.eval.load_hard_cube_states import load_cube_eval_states
from deepstatesearch.eval.solver import AStar, GreedyValueSolver
from deepstatesearch.model import ModelConfig, Model
from deepstatesearch.train.train import TrainConfig


@dataclass
class EvalConfig(DataStorage, json_name="eval_cfg.json", indent=4):
    solver:         str
    min_scrambles:  int
    max_scrambles:  int
    depths:         list[int]
    cube_data_file: str | None
    num_states:     int
    max_time:       float
    astar_lambda:   float
    astar_n:        int
    astar_d:        int
    solver_name:    str
    validate:       bool
    fp16:           bool

@dataclass
class EvalResults(DataStorage, json_name="eval_results.json", indent=4):
    num_solved:    int
    solved:        list[bool]
    solve_times:   list[float]
    states_seen:   list[int]
    solve_lengths: list[int | None]
    mem_usage:     list[int]

@torch.no_grad()
def eval(job: JobDescription):
    log.section("Loading configurations")
    eval_cfg = EvalConfig(
        solver         = job.solver,
        min_scrambles  = job.min_scrambles,
        max_scrambles  = job.max_scrambles,
        depths         = list(),
        cube_data_file = job.cube_data_file,
        num_states     = job.num_states,
        max_time       = job.max_time,
        astar_lambda   = job.astar_lambda,
        astar_n        = job.astar_n,
        astar_d        = job.astar_d,
        solver_name    = "",  # Set later
        validate       = job.validate,
        fp16           = job.fp16,
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
        TT.profile("Load models")
        log(f"Loading model {i+1} / {train_cfg.num_models}")
        model = Model(model_cfg).to(device)
        model.load_state_dict(torch.load(
            f"{job.location}/../model-{i}.pt",
            map_location=device,
        ))
        model.eval()
        models.append(model)
        TT.end_profile()

        with TT.profile("Warmup"):
            model(env.multiple_oh(env.get_multiple_solved(1).to(device)))
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if i == 0:
            log.debug(
                "Parameters per model: %s" % thousands_seperators(model.numel()),
                "Total parameters:     %s" % thousands_seperators(model.numel() * train_cfg.num_models),
            )
            log.debug(model)

    log.section("Preparing solver")
    if eval_cfg.solver == "GreedyValueSolver":
        solver = GreedyValueSolver(env, eval_cfg.max_time, models, eval_cfg.fp16)
    elif eval_cfg.solver == "AStar":
        solver = AStar(
            env, eval_cfg.max_time, models,eval_cfg.fp16,
            eval_cfg.astar_lambda, eval_cfg.astar_n, eval_cfg.astar_d,
        )
    else:
        raise ValueError("Unknown solver '%s'" % eval_cfg.solver)
    eval_cfg.solver_name = str(solver)

    results = EvalResults(
        num_solved    = 0,
        solved        = list(),
        solve_times   = list(),
        states_seen   = list(),
        solve_lengths = list(),
        mem_usage     = list(),
    )

    log.section("Evaluating")
    if eval_cfg.cube_data_file:
        log("Loading cube data file %s" % eval_cfg.cube_data_file)
        states = load_cube_eval_states(eval_cfg.cube_data_file)
        eval_cfg.depths = [24] * len(states)
        eval_cfg.num_states = len(states)
    else:
        log("Generating random states")
        states, eval_cfg.depths = gen_eval_states(env, eval_cfg.num_states, eval_cfg.min_scrambles, eval_cfg.max_scrambles)
        preds = models[0](env.multiple_oh(states.to(device)))
        log.debug(
            "Predicted mean and std. of cost-to-go of evaluation states",
            "Mean: %.2f" % preds.mean(),
            "Std.: %.2f" % preds.std(),
        )
    log.debug("Evaluation depths: %s" % eval_cfg.depths)

    for i, state in enumerate(states):
        log.debug(
            "State %i / %i" % (i + 1, eval_cfg.num_states),
            state,
        )
        actions, time, states_seen = solver.solve(state)
        did_solve = actions is not None and time <= eval_cfg.max_time
        results.num_solved += did_solve
        results.solved.append(did_solve)
        results.solve_times.append(time)
        results.states_seen.append(states_seen)
        results.solve_lengths.append(len(actions) if did_solve else -1)
        results.mem_usage.append(psutil.Process(os.getpid()).memory_info().rss)

        log(
            "Solved: %s" % did_solve,
            "Length: %i" % len(actions) if did_solve else -1,
            "Time:   %.4f s" % time,
            "States: %s" % thousands_seperators(states_seen),
            "States per second: %s" % thousands_seperators(round(states_seen / time)),
        )

        if did_solve and eval_cfg.validate:
            TT.profile("Validate")
            new_state = state
            for action in actions:
                new_state = env.move(action.item(), new_state)
            if not env.is_solved(new_state):
                log.error(
                    "State %s was not solved" % state,
                    "Actions:     %s" % actions,
                    "Final state: %s" % new_state,
                )
                raise RuntimeError("Failed to solve state")
            TT.end_profile()

    log.section("Calculating solution statistics")
    log("Solved %i / %i = %.0f %%" % (
        results.num_solved,
        eval_cfg.num_states,
        100 * results.num_solved / eval_cfg.num_states,
    ))
    solved_states = torch.BoolTensor(results.solved)
    if eval_cfg.cube_data_file:
        num_optimally_solved = sum(sl == 24 for sl in results.solve_lengths)
        log(
            "Solved optimally: %i (%.2f %%)" % (num_optimally_solved, 100 * num_optimally_solved / eval_cfg.num_states),
            "Solved:           %i (%.2f %%)" % (results.num_solved, 100 * results.num_solved / eval_cfg.num_states),
            "Total states:     %i" % eval_cfg.num_states,
        )
    if solved_states.any():
        # Log statistics for solved states
        solve_times = torch.FloatTensor(results.solve_times)[solved_states]
        states_seen = torch.FloatTensor(results.states_seen)[solved_states]
        solve_lengths = torch.FloatTensor(results.solve_lengths)[solved_states]
        log(
            "For the solved states:",
            "Avg. solution time:      %.2f ms" % (1000 * solve_times.mean()),
            "Avg. states seen:        %s" % thousands_seperators(round(states_seen.mean().item())),
            "Avg. nodes per second:   %s" % thousands_seperators(round(states_seen.mean().item() / solve_times.mean().item())),
            "Avg. solution length:    %.2f" % solve_lengths.mean(),
            "Solution length 95 %% CI: +/- %.2f" % (z() * solve_lengths.std() / math.sqrt(len(solve_lengths))),
            sep="\n    ",
        )

    log.section("Saving")
    eval_cfg.save(job.location)
    results.save(job.location)
