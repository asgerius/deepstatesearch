from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil

if "NO_CUDA" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
import torch
from pelutils import log, DataStorage, TT, thousands_seperators

from deepspeedcube import device
from deepspeedcube.benchmarks import savedir
from deepspeedcube.envs import get_env
from deepspeedcube.envs.gen_states import gen_new_states


env = get_env("cube")
cuda = torch.cuda.is_available()

@dataclass
class CubeResultsCPU(DataStorage, json_name="cube_results_cpu.json", indent=4):
    n: list[int]
    move_time: list[list[float]]
    move_time_inplace: list[list[float]]

@dataclass
class CubeResultsCUDA(DataStorage, json_name="cube_results_cuda.json", indent=4):
    n: list[int]
    move_time: list[list[float]]
    move_time_inplace: list[list[float]]

def benchmark():
    log.section("Benchmarking cube environment", f"CUDA = {cuda}")
    reps = 3
    res = (CubeResultsCUDA if cuda else CubeResultsCPU)(
        n = np.logspace(0, 9 if cuda else 7, 20, dtype=int).tolist(),
        move_time = [list() for _ in range(reps)],
        move_time_inplace = [list() for _ in range(reps)],
    )

    for i in range(reps):
        log("Repetition %i / %i" % (i, reps))
        for n in res.n:
            log(f"Running for n = {thousands_seperators(n)}")
            log.debug("Generating random states and actions")
            states, _ = gen_new_states(env, n, 30)
            actions = env.action_space.repeat(ceil(n / len(env.action_space)))[:n].to(device)

            log.debug("Performing actions")
            TT.tick()
            states = env.multiple_moves(actions, states)
            res.move_time[i].append(TT.tock()/n)

            log.debug("Performing actions inplace")
            TT.tick()
            states = env.multiple_moves(actions, states, inplace=True)
            res.move_time_inplace[i].append(TT.tock()/n)

    log("Saving results")
    res.save(savedir)

def plot():
    res_cpu = CubeResultsCPU.load(savedir)
    res_cuda = CubeResultsCUDA.load(savedir)

    times_cpu          = np.array(res_cpu.move_time) * 1e9
    times_inplace_cpu  = np.array(res_cpu.move_time_inplace) * 1e9
    times_cuda         = np.array(res_cuda.move_time) * 1e9
    times_inplace_cuda = np.array(res_cuda.move_time_inplace) * 1e9

    with plots.Figure(f"{savedir}/cube.png"):
        plt.plot(res_cpu.n,  times_cpu.mean(axis=0),          "-o",  c=plots.tab_colours[0], label="CPU")
        plt.plot(res_cpu.n,  times_inplace_cpu.mean(axis=0),  "--o", c=plots.tab_colours[0], label="CPU, inplace")
        plt.plot(res_cuda.n, times_cuda.mean(axis=0),         "-o",  c=plots.tab_colours[1], label="CUDA")
        plt.plot(res_cuda.n, times_inplace_cuda.mean(axis=0), "--o", c=plots.tab_colours[1], label="CUDA, inplace")
        # breakpoint()

        plt.xscale("log")
        plt.yscale("log")

        plt.grid()
        plt.xlabel("Number of states")
        plt.ylabel("Avg. move time [ns]")
        plt.legend(loc=1)

if __name__ == "__main__":
    log.configure(f"{savedir}/cube.log")
    with log.log_errors:
        benchmark()
        plot()
