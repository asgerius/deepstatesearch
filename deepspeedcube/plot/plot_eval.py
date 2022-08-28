import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pelutils.ds.plots as plots
from pelutils import log

from deepspeedcube.eval.eval import EvalConfig, EvalResults



def plot_solve_pct(loc: str, cfg: EvalConfig, res: EvalResults):
    solved_frac = np.array(res.solved).mean(axis=1)

    with plots.Figure(f"{loc}/plots-eval/solve-pct.png"):
        plt.figure().gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.plot(100 * solved_frac, "--o")
        plt.title("Solve pct. for %s" % cfg.solver_name)
        plt.xlabel("Scrambles")
        plt.ylabel("Solved [%]")
        plt.ylim([-7, 107])
        plt.grid()

def plot_solve_pct_all(loc: str, cfgs: list[EvalConfig], ress: list[EvalResults]):
    with plots.Figure(f"{loc}/plots-eval/solve-pct.png"):
        plt.figure().gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        for cfg, res in zip(cfgs, ress):
            solved_frac = np.array(res.solved).mean(axis=1)
            plt.plot(100 * solved_frac, "--o", label=cfg.solver_name)
        plt.legend()
        plt.xlabel("Scrambles")
        plt.ylabel("Solved [%]")
        plt.ylim([-7, 107])
        plt.grid()

def plot_solve_rate_time(loc: str, cfg: EvalConfig, res: EvalResults):
    with plots.Figure(f"{loc}/plots-eval/solve-rate-time.png"):
        solved = np.array(res.solved)[-1]
        solve_times = np.array(res.solve_times)[-1, solved]
        solve_times = np.sort(solve_times)
        solved_frac = np.linspace(0, solved.mean(), 1 + len(solve_times))[1:]

        if solved.any():
            plt.plot(solve_times, 100 * solved_frac, "--o")

            plt.xlabel("Wall time [s]")
            plt.ylabel("Solved states [%]")
            plt.xlim([-0.07 * solve_times.max(), 1.07 * solve_times.max()])
            plt.ylim([-7, 107])
            plt.grid()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    parser.add_argument("evals", nargs="*")
    args = parser.parse_args()

    log.configure(f"{args.location}/plots-eval/plot-eval.log")

    configs = list()
    results = list()
    for eval in args.evals:
        log("Plotting %s" % eval)
        loc = f"{args.location}/{eval}"
        os.makedirs(f"{loc}/plots-eval", exist_ok=True)

        cfg = EvalConfig.load(loc)
        res = EvalResults.load(loc)
        configs.append(cfg)
        results.append(res)

        plot_solve_pct(loc, cfg, res)
        plot_solve_rate_time(loc, cfg, res)

    log("Plotting combined")
    plot_solve_pct_all(args.location, configs, results)
