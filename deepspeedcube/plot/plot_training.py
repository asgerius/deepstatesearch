from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from pelutils import log

from deepspeedcube.train.train import TrainConfig, TrainResults


def plot_loss(loc: str, cfg: TrainConfig, res: TrainResults):
    with plots.Figure(f"{loc}/plots-train/loss.png"):
        losses = np.array(res.losses)
        mean_loss = losses.mean(axis=0)
        plt.plot(mean_loss, c=plots.tab_colours[0])
        plt.plot(losses.T, alpha=0.5, c=plots.tab_colours[0])
        plt.yscale("log")
        plt.grid()
        plt.xlabel("Batch")
        plt.ylabel("MSE loss")

def plot_value_estimates(loc: str, cfg: TrainConfig, res: TrainResults):
    with plots.Figure(f"{loc}/plots-train/value-estimates.png"):
        if cfg.env == "cube":
            Ks = (1, 3, 5, 7, 10, 13, 15, 17, 20, 30, 40)
        else:
            # Generally space Ks quadratically, as log spacing is too agressive
            Ks = np.unique((np.linspace(1, np.sqrt(cfg.K), 10) ** 2).to(int))
        for K in Ks:
            if K > cfg.K:
                break
            plt.plot(res.eval_idx, res.value_estimations[K-1], label="$K=%i$" % K)

        plt.grid()
        plt.xlabel("Batch")
        plt.ylabel("Avg. value estimate")
        plt.legend(loc="upper left")

def plot_lr(loc: str, cfg: TrainConfig, res: TrainResults):
    with plots.Figure(f"{loc}/plots-train/lr.png"):
        plt.plot(np.arange(cfg.batches), res.lr)
        plt.grid()
        plt.xlabel("Batch")
        plt.ylabel("Learning rate")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("location")
    args = parser.parse_args()
    log.configure(f"{args.location}/plots-train/plot-training.log")

    with log.log_errors:
        log("Loading config and results")
        cfg = TrainConfig.load(args.location)
        res = TrainResults.load(args.location)

        log("Plotting loss")
        plot_loss(args.location, cfg, res)
        log("Plotting value estimates")
        plot_value_estimates(args.location, cfg, res)
        log("Plotting learning rate")
        plot_lr(args.location, cfg, res)
