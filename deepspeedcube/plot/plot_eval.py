import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pelutils.ds.plots as plots
from pelutils import log

from deepspeedcube.eval.eval import EvalConfig, EvalResults


def plot_solve_rate_time(loc: str, cfg: EvalConfig, res: EvalResults):
    with plots.Figure(f"{loc}/plots-eval/solve-rate-time.png"):
        solved = np.array(res.solved)
        solve_times = np.array(res.solve_times)[solved]
        solve_times = np.sort(solve_times)
        solved_frac = np.linspace(0, solved.mean(), 1 + len(solve_times))[1:]

        if solved.any():
            plt.plot(solve_times, 100 * solved_frac, "--o")

            plt.xlabel("Wall time [s]")
            plt.ylabel("Solved states [%]")
            plt.xlim([-0.07 * solve_times.max(), 1.07 * solve_times.max()])
            plt.ylim([-7, 107])
            plt.grid()

def plot_solve_states_seen(loc: str, cfg: EvalConfig, res: EvalResults):
    with plots.Figure(f"{loc}/plots-eval/solve-rate-states.png"):
        solved = np.array(res.solved)
        states_seen = np.array(res.states_seen)[solved]
        states_seen = np.sort(states_seen)
        solved_frac = np.linspace(0, solved.mean(), 1 + len(states_seen))[1:]

        if solved.any():
            plt.plot(states_seen, 100 * solved_frac, "--o")

            plt.xlabel("States seen during search")
            plt.ylabel("Solved states [%]")
            plt.xlim([-0.07 * states_seen.max(), 1.07 * states_seen.max()])
            plt.ylim([-7, 107])
            plt.grid()

def plot_states_seen(loc: str, cfg: EvalConfig, res: EvalResults):
    with plots.Figure(f"{loc}/plots-eval/states-seen.png"):
        solved = np.array(res.solved)
        solve_times = np.array(res.solve_times)
        states_seen = np.array(res.states_seen)

        for did_solve in True, False:
            plt.scatter(solve_times[solved==did_solve], states_seen[solved==did_solve], label="Solved" if did_solve else "Not solved")

        plt.legend()
        plt.xlabel("Wall time [s]")
        plt.ylabel("States seen")
        plt.grid()

def plot_memory_usage(loc: str, cfg: EvalConfig, res: EvalResults):
    with plots.Figure(f"{loc}/plots-eval/memory-usage.png"):
        memory_usage = np.array(res.mem_usage).ravel()

        plt.plot(memory_usage / 2 ** 20, "-o")
        plt.title("Memory usage during evaluation")
        plt.xlabel("Evaluated states")
        plt.ylabel("Memory usage [MB]")
        plt.grid()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    parser.add_argument("evals", nargs="*")
    args = parser.parse_args()

    log.configure(f"{args.location}/plots-eval/plot-eval.log")

    configs = list()
    results = list()
    for eval_ in args.evals:
        log("Plotting %s" % eval_)
        loc = f"{args.location}/{eval_}"
        os.makedirs(f"{loc}/plots-eval", exist_ok=True)

        cfg = EvalConfig.load(loc)
        res = EvalResults.load(loc)
        configs.append(cfg)
        results.append(res)

        plot_solve_rate_time(loc, cfg, res)
        plot_solve_states_seen(loc, cfg, res)
        plot_states_seen(loc, cfg, res)
        plot_memory_usage(loc, cfg, res)
