from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
import torch
from pelutils import log, DataStorage, TT, thousands_seperators

from deepstatesearch import device, HardwareInfo
from deepstatesearch.benchmarks import savedir
from deepstatesearch.envs import get_env
from deepstatesearch.envs.gen_states import gen_new_states


cuda = torch.cuda.is_available()
device_name = HardwareInfo.gpu or HardwareInfo.cpu

@dataclass
class CubeResults(DataStorage, json_name="cube_results.json", indent=4):
    n: list[int]
    scramble_time:     list[list[float]]
    move_time:         list[list[float]]
    move_time_inplace: list[list[float]]

    cpu:     str = HardwareInfo.cpu
    sockets: int = HardwareInfo.sockets
    cores:   int = HardwareInfo.cores
    gpu:     str = HardwareInfo.gpu

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

    reps = 20
    res = CubeResults(
        n = np.logspace(0, 8.5, 20, dtype=int).tolist(),
        scramble_time     = [list() for _ in range(reps)],
        move_time         = [list() for _ in range(reps)],
        move_time_inplace = [list() for _ in range(reps)],
    )

    for i in range(reps):
        log.section("Repetition %i / %i" % (i+1, reps))
        for n in res.n:
            log(f"Running for n = {thousands_seperators(n)}")
            TT.tick()
            states, _ = gen_new_states(env, n, 30)
            t = TT.tock()
            res.scramble_time[i].append(t/n)

            log.debug("Scramble: %.6f ms" % (1e3*t))

            actions = env.action_space.repeat(ceil(n / len(env.action_space)))[:n].to(device)

            TT.tick()
            states = env.multiple_moves(actions, states)
            t = TT.tock()
            log.debug("OOP:      %.6f ms" % (1e3*t))
            res.move_time[i].append(t/n)

            TT.tick()
            states = env.multiple_moves(actions, states, inplace=True)
            t = TT.tock()
            log.debug("Inplace:  %.6f ms" % (1e3*t))
            res.move_time_inplace[i].append(t/n)

    log("Saving results")
    res.save(f"{savedir}/cube_{device_name}")

def plot():
    names = [x for x in os.listdir(savedir) if os.path.isdir(f"{savedir}/{x}") and x.startswith("cube_")]
    res = [CubeResults.load(f"{savedir}/{name}") for name in names]
    names = [name.removeprefix("cube_") for name in names]
    times = [1e9 * np.array(r.move_time) for r in res]
    times_inplace = [1e9 * np.array(r.move_time_inplace) for r in res]

    with plots.Figure(f"{savedir}/cube-move.png", legend_fontsize=0.75):
        for i, r in enumerate(res):
            times = 1e9 * np.array(r.move_time).mean(axis=0)
            times_inplace = 1e9 * np.array(r.move_time_inplace).mean(axis=0)

            if r.gpu is None:
                name = f"{r.cores//r.sockets} C " + names[i]
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

    with plots.Figure(f"{savedir}/cube-gen.png", legend_fontsize=0.75):
        for i, r in enumerate(res):
            times = 1e9 * np.array(r.scramble_time).mean(axis=0)

            if r.gpu is None:
                name = f"{r.cores//r.sockets} C " + names[i]
            else:
                name = names[i]
            plt.plot(r.n, times,         "-o",  c=plots.tab_colours[i], label=name)

        plt.xscale("log")
        plt.yscale("log")

        plt.grid()
        plt.xlabel("Number of states")
        plt.ylabel("Avg. gen. time [ns]")
        plt.legend(loc=1)

if __name__ == "__main__":
    with log.log_errors:
        parser = ArgumentParser()
        parser.add_argument("--plot", action="store_true")
        parser.add_argument("--env", default="cube")
        args = parser.parse_args()
        env = get_env(args.env)
        if args.plot:
            plot()
        else:
            benchmark()
