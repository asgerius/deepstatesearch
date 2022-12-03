from __future__ import annotations

import os

from pelutils import log, TT, LogLevels
from pelutils.parser import Parser, Flag, Option

from deepstatesearch import HardwareInfo
from deepstatesearch.eval.eval import eval


options = (
    Option("min-scrambles",  default=1000, type=int),
    Option("max-scrambles",  default=10000, type=int),
    Option("num-states",     default=100, type=int),
    Option("cube-data-file", default=None, type=str, help="Path to file with scrambles. Overrides min-scrambles, max-scrambles, and num-states."),
    Option("solver",         default="AStar"),
    Option("max-time",       default=1, type=float),
    Option("astar-lambda",   default=1, type=float),
    Option("astar-n",        default=100),
    Option("astar-d",        default=2),
    Flag("validate"),
    Flag("fp16"),
)

if __name__ == "__main__":
    parser = Parser(*options, multiple_jobs=True)
    jobs = parser.parse_args(clear_folders=True)

    for i, job in enumerate(jobs):
        TT.reset()
        log.configure(f"{job.location}/eval.log", print_level=LogLevels.DEBUG)
        with log.log_errors:
            log.section(f"Evaluating {i+1} / {len(jobs)}: {job.name}")
            log.log_repo()
            log(
                "Hardware info:",
                "CPU:     %s" % HardwareInfo.cpu,
                "Sockets: %i" % HardwareInfo.sockets,
                "Cores:   %i" % HardwareInfo.cores,
                "GPU:     %s" % HardwareInfo.gpu,
                sep="\n    ",
            )
            log("OMP_NUM_THREADS = %s" % os.environ.get("OMP_NUM_THREADS"), with_info=False)
            eval(job)
            log.debug("Time distribution", TT)
