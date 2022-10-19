from __future__ import annotations

from dataclasses import dataclass, field

from pelutils import DataStorage


@dataclass
class TrainConfig(DataStorage, json_name="train_config.json", indent=4):
    env:                str
    num_models:         int
    batches:            int
    batch_size:         int
    K:                  int
    lr:                 float
    tau:                float
    tau_every:          int
    weight_decay:       float
    epsilon:            float
    known_states_depth: int
    fp16:               bool

@dataclass
class TrainResults(DataStorage, json_name="train_results.json", indent=4):
    current_batch: int
    eval_idx: list[int] = field(default_factory=list)
    # depth [ batch [ value ] ]
    value_estimations: list[list[float]] = field(default_factory=list)

    lr: list[float] = field(default_factory=list)
    # model [ batch [ loss ] ]
    losses: list[list[float]] = field(default_factory=list)
