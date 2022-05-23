from __future__ import annotations

from pprint import pformat

from pelutils import log
from pelutils.parser import JobDescription


def train(job: JobDescription):
    log("Job arguments", pformat(job.todict()))
