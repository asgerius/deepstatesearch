from __future__ import annotations

from pelutils import log, TT
from pelutils.parser import Option, Parser

from deepspeedcube.train.train import train

options = (
    Option("env",                 default="cube"),
    Option("batches",             default=100),
    Option("batch-size",          default=1000),
    Option("scramble_depth",      default=30),
    Option("lr",                  default=1e-4),
    Option("num-models",          default=1),
    Option("tau",                 default=0.3, help="1 for no generator network and 0 for static generator network"),
    Option("j-norm",              default=20, type=float),
    Option("hidden-layer-sizes",  default=[4096, 1024], type=int, nargs=0),
    Option("num-residual-blocks", default=4),
    Option("residual-size",       default=1024),
    Option("dropout",             default=0),
)

if __name__ == "__main__":
    parser = Parser(*options, multiple_jobs=True)
    job_descriptions = parser.parse_args()
    for i, job in enumerate(job_descriptions):
        TT.reset()
        log.configure(f"{job.location}/train.log")
        with log.log_errors:
            log.section(f"Training {i+1} / {len(job_descriptions)}")
            train(job)
            log("Time distribution", TT)

    parser.document()
