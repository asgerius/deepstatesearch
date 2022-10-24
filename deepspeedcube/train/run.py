from __future__ import annotations

import os

from pelutils import log, TT
from pelutils.parser import Flag, Option, Parser

from deepspeedcube import HardwareInfo
from deepspeedcube.train.train import train


options = (
    Option("env",                 default="cube"),
    Option("batches",             default=10000),
    Option("batch-size",          default=1000),
    Option("k",                   default=30),
    Option("lr",                  default=1e-5),
    Option("num-models",          default=1),
    Option("tau",                 default=0.1, help="1 for no generator network and 0 for static generator network"),
    Option("tau-every",           default=500),
    Option("j-norm",              default=1, type=float),
    Option("hidden-layer-sizes",  default=[4096, 1024], type=int, nargs=0),
    Option("num-residual-blocks", default=4),
    Option("residual-size",       default=1024),
    Option("dropout",             default=0, type=float),
    Option("weight-decay",        default=0, type=float),
    Option("epsilon",             default=1e6, type=float),
    Option("known-states-depth",  default=0),
    Flag("fp16"),
    Flag("resume"),
)

if __name__ == "__main__":
    parser = Parser(*options, multiple_jobs=True)
    job_descriptions = parser.parse_args()
    for i, job in enumerate(job_descriptions):
        TT.reset()
        log.configure(f"{job.location}/train.log", append=job.resume)
        with log.log_errors:
            log.section(f"Training {i+1} / {len(job_descriptions)}: {job.name}")
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
            train(job)

    parser.document()
