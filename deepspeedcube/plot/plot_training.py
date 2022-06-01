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
        plt.grid()
        plt.xlabel("Batch")
        plt.ylabel("MSE Loss")

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
        log("Plotting learning rate")
        plot_lr(args.location, cfg, res)
