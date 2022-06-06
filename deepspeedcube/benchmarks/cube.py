from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil

if "NO_CUDA" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
import torch
from pelutils import log, DataStorage, TT, thousands_seperators, EnvVars

from deepspeedcube import device, HardwareInfo
from deepspeedcube.benchmarks import savedir
from deepspeedcube.envs import get_env
from deepspeedcube.envs.gen_states import gen_new_states


env = get_env("cube")
cuda = torch.cuda.is_available()
device_name = HardwareInfo.gpu or HardwareInfo.cpu

@dataclass
class CubeResults(DataStorage, json_name="cube_results.json", indent=4):
    n: list[int]
    move_time: list[list[float]]
    move_time_inplace: list[list[float]]

    cpu     = HardwareInfo.cpu
    sockets = HardwareInfo.sockets
    cores   = HardwareInfo.cores
    gpu     = HardwareInfo.gpu

def benchmark():
    log.configure(f"{savedir}/cube_{device_name}.log")

    log.section("Benchmarking cube environment", f"CUDA = {cuda}")

    log(
        "Hardware info:",
        "CPU:     %s" % HardwareInfo.cpu,
        "Sockets: %i" % HardwareInfo.sockets,
        "Cores:   %i" % HardwareInfo.cores,
        "GPU:     %s" % HardwareInfo.gpu,
        sep="\n    ",
    )

    reps = 5
    res = CubeResults(
        n = np.logspace(0, 9 if cuda else 8, 20, dtype=int).tolist(),
        move_time = [list() for _ in range(reps)],
        move_time_inplace = [list() for _ in range(reps)],
    )

    for i in range(reps):
        log.section("Repetition %i / %i" % (i+1, reps))
        for n in res.n:
            log(f"Running for n = {thousands_seperators(n)}")
            log.debug("Generating random states and actions")
            states, _ = gen_new_states(env, n, 30)
            actions = env.action_space.repeat(ceil(n / len(env.action_space)))[:n].to(device)

            log.debug("Performing actions")
            TT.tick()
            states = env.multiple_moves(actions, states)
            t = TT.tock()
            log.debug("Used %.6f ms" % (1e3*t))
            res.move_time[i].append(t/n)

            log.debug("Performing actions inplace")
            TT.tick()
            states = env.multiple_moves(actions, states, inplace=True)
            t = TT.tock()
            log.debug("Used %.6f ms" % (1e3*t))
            res.move_time_inplace[i].append(t/n)

    log("Saving results")
    res.save(f"{savedir}/cube_{device_name}")

def plot():
    names = [x for x in os.listdir(savedir) if os.path.isdir(f"{savedir}/{x}") and x.startswith("cube_")]
    res = [CubeResults.load(f"{savedir}/{name}") for name in names]
    names = [name.removeprefix("cube_") for name in names]
    times = [1e9 * np.array(r.move_time) for r in res]
    times_inplace = [1e9 * np.array(r.move_time_inplace) for r in res]

    with plots.Figure(f"{savedir}/cube.png", legend_fontsize=0.75):
        for i, r in enumerate(res):
            times = 1e9 * np.array(r.move_time).mean(axis=0)
            times_inplace = 1e9 * np.array(r.move_time_inplace).mean(axis=0)

            if r.gpu is None:
                name = f"{r.sockets} x {r.cores//r.sockets} C " + names[i]
            else:
                name = names[i]
            plt.plot(r.n, times,         "-o",  c=plots.tab_colours[i], label=name)
            plt.plot(r.n, times_inplace, "--o", c=plots.tab_colours[i], label=f"{name}, inplace")

        plt.xscale("log")
        plt.yscale("log")

        plt.grid()
        plt.xlabel("Number of states")
        plt.ylabel("Avg. move time [ns]")
        plt.legend(loc=1)

if __name__ == "__main__":
    with log.log_errors:
        parser = ArgumentParser()
        parser.add_argument("--plot", action="store_true")
        args = parser.parse_args()
        if args.plot:
            plot()
        else:
            benchmark()
