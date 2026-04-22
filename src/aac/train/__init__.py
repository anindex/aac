"""Training pipeline: gap-closing loss, query pair dataset, and compressor trainer."""

from aac.train.data import QueryPairDataset, make_splits
from aac.train.loss import gap_closing_loss
from aac.train.trainer import TrainConfig, train_compressor

__all__ = [
    "gap_closing_loss",
    "QueryPairDataset",
    "make_splits",
    "TrainConfig",
    "train_compressor",
]
