from enum import Enum


class Monitor(Enum):
    MEAN_RANK = "mr"
    FILTERED_MEAN_RANK = "fmr"
    MEAN_RECIPROCAL_RANK = "mrr"
    FILTERED_MEAN_RECIPROCAL_RANK = "fmrr"


class TrainingStrategy(Enum):
    PROJECTION_BASED = "projection_based"   # matching models with neural network
    PAIRWISE_BASED = "pairwise_based"       # translational distance models
    POINTWISE_BASED = "pointwise_based"     # semantic matching models