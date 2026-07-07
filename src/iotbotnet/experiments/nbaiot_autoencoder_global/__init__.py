"""Experimentos de autoencoder global sobre N-BaIoT."""

from .stage_1_train_screening import run_nbaiot_autoencoder_global_stage_1
from .stage_2_train_optimize import run_nbaiot_autoencoder_global_stage_2
from .stage_3_evaluate import run_nbaiot_autoencoder_global_stage_3

__all__ = [
    "run_nbaiot_autoencoder_global_stage_1",
    "run_nbaiot_autoencoder_global_stage_2",
    "run_nbaiot_autoencoder_global_stage_3",
]
