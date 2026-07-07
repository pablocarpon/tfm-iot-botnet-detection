from __future__ import annotations

import gc
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
import yaml

from iotbotnet.data.datasets import (
    make_autoencoder_dataset_from_parquet,
    make_tf_dataset_from_parquet,
)
from iotbotnet.models.factory import build_model

LOGGER = logging.getLogger(__name__)


def run_nbaiot_autoencoder_global_stage_1(config_path: str | Path) -> dict[str, Any]:
    """Ejecuta el Stage 1 del autoencoder global sobre N-BaIoT.

    Para cada configuración preliminar:
    1. Entrena un único autoencoder con train benigno global.
    2. Monitoriza val_loss sobre validación benigna global.
    3. Guarda el modelo final, el histórico de entrenamiento y los errores de reconstrucción
       de train/val para su análisis posterior en notebook.

    Esta etapa NO selecciona thresholds y NO evalúa sobre test.
    """
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    _configure_logging(config)

    seed = int(config["experiment"].get("seed", 42))
    _set_reproducibility(seed)
    _configure_tensorflow_runtime(config)

    experiment_name = str(config["experiment"]["name"])
    data_dir = Path(config["paths"]["data_dir"])
    output_dir = Path(config["outputs"]["output_dir"])
    overwrite = bool(config["outputs"].get("overwrite", False))

    _prepare_output_dir(output_dir=output_dir, overwrite=overwrite)
    shutil.copy2(config_path, output_dir / "config.yaml")

    model_configs = list(config["model_configs"])
    metadata_columns = list(config["data"].get("metadata_columns", []))
    train_file = str(config["data"].get("train_file", "train.parquet"))
    val_file = str(config["data"].get("val_file", "val.parquet"))

    train_path = data_dir / train_file
    val_path = data_dir / val_file

    _validate_input_file(train_path)
    _validate_input_file(val_path)

    feature_columns = _get_feature_columns(train_path, metadata_columns)

    training_config = dict(config["training"])
    batch_size = int(training_config.get("batch_size", 2048))
    epochs = int(training_config.get("epochs", 100))
    verbose = int(training_config.get("verbose", 1))
    shuffle_train = bool(training_config.get("shuffle_train", False))
    shuffle_buffer_size = int(training_config.get("shuffle_buffer_size", 50_000))

    global_summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "dataset": config["experiment"].get("dataset"),
        "strategy": config["experiment"].get("strategy"),
        "seed": seed,
        "stage": 1,
        "n_features": len(feature_columns),
        "n_model_configs": len(model_configs),
        "n_trainings": len(model_configs),
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "metadata_columns": metadata_columns,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "shuffle_train": shuffle_train,
            "early_stopping": training_config.get("early_stopping", {}),
        },
        "configs": {},
    }

    LOGGER.info("Starting experiment: %s", experiment_name)
    LOGGER.info("Data directory: %s", data_dir)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Model configs: %d", len(model_configs))
    LOGGER.info("Feature columns: %d", len(feature_columns))

    for model_config in model_configs:
        config_id = str(model_config["id"])
        LOGGER.info("[%s] Starting training", config_id)

        run_output_dir = output_dir / "configs" / config_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        final_model_config = _build_model_config(
            model_config=model_config,
            input_dim=len(feature_columns),
        )

        _save_json(run_output_dir / "model_config.json", final_model_config)

        train_dataset = make_autoencoder_dataset_from_parquet(
            parquet_file=train_path,
            feature_columns=feature_columns,
            batch_size=batch_size,
            shuffle=shuffle_train,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        val_dataset = make_autoencoder_dataset_from_parquet(
            parquet_file=val_path,
            feature_columns=feature_columns,
            batch_size=batch_size,
            shuffle=False,
        )

        model = build_model(final_model_config)
        callbacks = _build_callbacks(training_config)

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

        history_df = pd.DataFrame(history.history)
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
        history_df.to_csv(run_output_dir / "history.csv", index=False)

        model.save(run_output_dir / "model")

        train_score_dataset = make_tf_dataset_from_parquet(
            parquet_file=train_path,
            feature_columns=feature_columns,
            batch_size=batch_size,
            shuffle=False,
        )
        val_score_dataset = make_tf_dataset_from_parquet(
            parquet_file=val_path,
            feature_columns=feature_columns,
            batch_size=batch_size,
            shuffle=False,
        )

        LOGGER.info("[%s] Scoring train split", config_id)
        train_scores = model.score_samples(train_score_dataset)
        _save_scores_with_metadata(
            scores=train_scores,
            source_parquet=train_path,
            metadata_columns=metadata_columns,
            output_path=run_output_dir / "train_reconstruction_errors.parquet",
        )

        LOGGER.info("[%s] Scoring validation split", config_id)
        val_scores = model.score_samples(val_score_dataset)
        _save_scores_with_metadata(
            scores=val_scores,
            source_parquet=val_path,
            metadata_columns=metadata_columns,
            output_path=run_output_dir / "val_reconstruction_errors.parquet",
        )

        best_epoch, best_val_loss = _get_best_epoch_and_val_loss(history_df)

        run_summary = {
            "model_name": model_config.get("name", config_id),
            "model_config": final_model_config,
            "epochs_trained": int(len(history_df)),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_error_summary": _summarize_scores(train_scores),
            "val_error_summary": _summarize_scores(val_scores),
        }

        _save_json(run_output_dir / "summary.json", run_summary)
        global_summary["configs"][config_id] = run_summary
        _save_json(output_dir / "global_summary.json", global_summary)

        del (
            model,
            train_dataset,
            val_dataset,
            train_score_dataset,
            val_score_dataset,
            train_scores,
            val_scores,
            history_df,
        )
        tf.keras.backend.clear_session()
        gc.collect()

        LOGGER.info("[%s] Finished", config_id)

    LOGGER.info("Experiment finished successfully")
    return global_summary


def _build_model_config(model_config: dict[str, Any], input_dim: int) -> dict[str, Any]:
    """Construye la configuración final del autoencoder a partir de una entrada YAML."""
    final_model_config = {
        "type": "autoencoder",
        "input_dim": input_dim,
    }

    params = dict(model_config.get("params", {}))
    final_model_config.update(params)

    if "reconstruction_error" not in final_model_config:
        final_model_config["reconstruction_error"] = final_model_config.get("loss", "mse")

    return final_model_config


def _build_callbacks(training_config: dict[str, Any]) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = [tf.keras.callbacks.TerminateOnNaN()]

    early_stopping_config = dict(training_config.get("early_stopping", {}))
    if early_stopping_config.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=str(early_stopping_config.get("monitor", "val_loss")),
                patience=int(early_stopping_config.get("patience", 10)),
                min_delta=float(early_stopping_config.get("min_delta", 1e-5)),
                restore_best_weights=bool(
                    early_stopping_config.get("restore_best_weights", True)
                ),
                mode=str(early_stopping_config.get("mode", "min")),
            )
        )

    return callbacks


def _get_best_epoch_and_val_loss(history_df: pd.DataFrame) -> tuple[int | None, float | None]:
    if "val_loss" not in history_df.columns or history_df.empty:
        return None, None

    best_idx = int(history_df["val_loss"].idxmin())
    return int(history_df.loc[best_idx, "epoch"]), float(history_df.loc[best_idx, "val_loss"])


def _summarize_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p50": float(np.quantile(scores, 0.50)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
    }


def _save_scores_with_metadata(
    scores: np.ndarray,
    source_parquet: Path,
    metadata_columns: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available_metadata_columns = _get_existing_columns(source_parquet, metadata_columns)
    if available_metadata_columns:
        scores_df = pd.read_parquet(source_parquet, columns=available_metadata_columns)
        if len(scores_df) != len(scores):
            raise ValueError(
                f"Scores length ({len(scores)}) does not match metadata length "
                f"({len(scores_df)}) for {source_parquet}."
            )
        scores_df.insert(0, "reconstruction_error", scores.astype(np.float32, copy=False))
    else:
        scores_df = pd.DataFrame(
            {"reconstruction_error": scores.astype(np.float32, copy=False)}
        )

    scores_df.to_parquet(output_path, index=False)


def _get_existing_columns(parquet_path: Path, candidate_columns: list[str]) -> list[str]:
    schema = pq.read_schema(parquet_path)
    columns = set(schema.names)
    return [col for col in candidate_columns if col in columns]


def _get_feature_columns(parquet_path: Path, metadata_columns: list[str]) -> list[str]:
    schema = pq.read_schema(parquet_path)
    columns = list(schema.names)

    metadata = set(metadata_columns)
    feature_columns = [col for col in columns if col not in metadata]

    if not feature_columns:
        raise ValueError(f"No feature columns found in {parquet_path}.")

    return feature_columns


def _set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not enable TensorFlow op determinism: %s", exc)


def _configure_tensorflow_runtime(config: dict[str, Any]) -> None:
    runtime_config = dict(config.get("runtime", {}))
    memory_growth = bool(runtime_config.get("gpu_memory_growth", True))

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        LOGGER.warning("No GPU detected by TensorFlow. Training will run on CPU.")
        return

    LOGGER.info("TensorFlow detected %d GPU(s): %s", len(gpus), gpus)

    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                LOGGER.warning("Could not set memory growth for %s: %s", gpu, exc)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Set outputs.overwrite=true in the config to overwrite it."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")

    return config


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _configure_logging(config: dict[str, Any]) -> None:
    level_name = str(config.get("runtime", {}).get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )