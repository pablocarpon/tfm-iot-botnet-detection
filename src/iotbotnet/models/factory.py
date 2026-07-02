from __future__ import annotations

from typing import Any

from iotbotnet.models.autoencoder import AutoencoderDetector
from iotbotnet.models.lof import LOFAnomalyDetector


def build_model(model_config: dict[str, Any]) -> AutoencoderDetector | LOFAnomalyDetector:
    """Construye un detector de anomalías a partir de la configuración del modelo.

    La factory solo interpreta la sección `model` de la configuración.
    No gestiona entrenamiento, callbacks, datasets, thresholds ni evaluación.
    """
    model_type = model_config.get("type")

    if model_type == "lof":
        return _build_lof(model_config)

    if model_type == "autoencoder":
        return _build_autoencoder(model_config)

    raise ValueError(
        f"Unsupported model type: {model_type!r}. "
        "Supported values are: 'lof', 'autoencoder'."
    )


def _build_lof(model_config: dict[str, Any]) -> LOFAnomalyDetector:
    return LOFAnomalyDetector(
        n_neighbors=model_config.get("n_neighbors", 20),
        metric=model_config.get("metric", "minkowski"),
        n_jobs=model_config.get("n_jobs", -1),
    )


def _build_autoencoder(model_config: dict[str, Any]) -> AutoencoderDetector:
    if "input_dim" not in model_config:
        raise ValueError("Missing required parameter for autoencoder: 'input_dim'.")

    return AutoencoderDetector(
        input_dim=model_config["input_dim"],
        latent_dim=model_config.get("latent_dim", 16),
        hidden_dims=model_config.get("hidden_dims"),
        activation=model_config.get("activation", "relu"),
        output_activation=model_config.get("output_activation", "linear"),
        dropout_rate=model_config.get("dropout_rate", 0.0),
        batch_norm=model_config.get("batch_norm", False),
        optimizer=model_config.get("optimizer", "adam"),
        learning_rate=model_config.get("learning_rate", 1e-3),
        loss=model_config.get("loss", "mse"),
        reconstruction_error=model_config.get("reconstruction_error", "mse"),
    )