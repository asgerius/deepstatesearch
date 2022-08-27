from __future__ import annotations

from pelutils import log, TT
from pelutils.parser import Parser, Flag, Option

from deepspeedcube import HardwareInfo
from deepspeedcube.eval.eval import eval


options = (
    Option("max-depth",        default=30, type=int),
    Option("states-per-depth", default=100, type=int),
    Option("solver",           default="AStar"),
    Option("max-time",         default=1, type=float),
    Option("astar-lambda",     default=1, type=float),
    Option("astar-n",          default=100),
    Option("astar-d",          default=1),
    Flag("validate"),
)

if __name__ == "__main__":
    parser = Parser(*options, multiple_jobs=True)
    jobs = parser.parse_args(clear_folders=True)

    for i, job in enumerate(jobs):
        TT.reset()
        log.configure(f"{job.location}/eval.log")
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
            eval(job)
            log.debug("Time distribution", TT)
