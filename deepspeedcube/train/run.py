from __future__ import annotations

import pelutils
from pelutils import log, TickTock
from pelutils.parser import Flag, Option, Parser

from deepspeedcube.train.train import train

options = (
    Option("env", default="cube"),
    Option("hidden-layers", abbrv="H", default=[4096, 2048, 512], type=int, nargs=0),
    Flag("fp16"),
)

if __name__ == "__main__":
    parser = Parser(*options, multiple_jobs=True)
    job_descriptions = parser.parse_args()
    for i, job in enumerate(job_descriptions):
        # Reset global tick tock instance for new job
        pelutils.TT = TickTock()
        log.configure(f"{job.location}/train.log")
        with log.log_errors:
            log.section(f"Training {i+1} / {len(job_descriptions)}")
            train(job)
            log.debug("Time distribution", pelutils.TT)

    parser.document()
