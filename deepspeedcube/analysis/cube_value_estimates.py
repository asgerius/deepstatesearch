from __future__ import annotations

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
import torch
from tqdm import tqdm

from deepspeedcube import device
from deepspeedcube.envs import get_env
from deepspeedcube.eval.load_hard_cube_states import load_hard_and_intermediate_states
from deepspeedcube.model import Model, ModelConfig
from deepspeedcube.train.train import TrainConfig


env = get_env("cube")

@torch.no_grad()
def value_estimates(out: str, qtm_datafile: str, model_names: list[str], model_sets: list[list[Model]]):

    states = load_hard_and_intermediate_states(qtm_datafile).view(-1, *env.state_shape)
    states_d = states.to(device)
    states_oh = env.multiple_oh(states_d)
    states = states.view(25, -1, *env.state_shape)
    states_oh = states_oh.view(25, -1, env.state_oh_size)

    with plots.Figure(f"{out}/value-estimates-24.png"):
        for model_name, model_set in tqdm(zip(model_names, model_sets), total=len(model_sets)):
            preds = np.zeros(states.shape[1])
            for model in model_set:
                preds += model(states_oh[-1]).squeeze().cpu().numpy()
            preds /= len(model_set)

            plt.plot(*plots.get_bins(preds, plots.normal_binning, bins=50), "-o", label=model_name)

        plt.legend()
        plt.xlabel("$J$")
        plt.ylabel("Probability density")
        plt.grid()

    with plots.Figure(f"{out}/value-estimates.png"):
        plt.plot(np.arange(25), np.arange(25), "-o", color="grey", label="True value", alpha=0.7)

        for model_name, model_set in tqdm(zip(model_names, model_sets), total=len(model_sets)):
            preds = np.zeros((25, states.shape[1]))
            for j in range(25):
                for model in model_set:
                    preds[j] += model(states_oh[j]).squeeze().cpu().numpy()
            preds /= len(model_set)

            plt.plot(preds.mean(axis=1), "-o", label=model_name)

        plt.legend()
        plt.xlabel("$J$")
        plt.ylabel("True shortest distance")
        plt.grid()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("qtm")
    parser.add_argument("-m", "--model-dirs", nargs="+")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    model_sets = list()
    for model_dir in args.model_dirs:
        train_cfg = TrainConfig.load(model_dir)
        model_cfg = ModelConfig.load(model_dir)
        model_set = list()
        for i in range(train_cfg.num_models):
            model = Model(model_cfg).to(device)
            model.load_state_dict(torch.load(
                f"{model_dir}/model-{i}.pt",
                map_location=device,
            ))
            model.eval()
            model_set.append(model)
        model_sets.append(model_set)

    value_estimates(
        args.out,
        args.qtm,
        [os.path.split(x)[-1] for x in args.model_dirs],
        model_sets,
    )
