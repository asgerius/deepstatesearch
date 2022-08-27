import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from pelutils import log

from deepspeedcube.eval.eval import EvalConfig, EvalResults


def plot_solve_pct(loc: str, cfg: EvalConfig, res: EvalResults):
    solved_frac = np.array(res.solved) / cfg.states_per_depth

    with plots.Figure(f"{loc}/plots-eval/solve-pct.png"):
        plt.plot(100 * solved_frac, "--o")
        plt.title("Solve pct. for %s" % cfg.solver_name)
        plt.xlabel("Scrambles")
        plt.ylabel("Solved [%]")
        plt.grid()

def plot_solve_pct_all(loc: str, evals: list[str], cfgs: list[EvalConfig], ress: list[EvalResults]):
    with plots.Figure(f"{loc}/plots-eval/solve-pct.png"):
        for eval, cfg, res in zip(evals, cfgs, ress):
            solved_frac = np.array(res.solved) / cfg.states_per_depth
            plt.plot(100 * solved_frac, "--o", label=cfg.solver_name)
        plt.xlabel("Scrambles")
        plt.ylabel("Solved [%]")
        plt.grid()
        plt.legend()

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

    log("Plotting combined")
    plot_solve_pct_all(args.location, args.evals, configs, results)
