from __future__ import annotations

import gc
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from iotbotnet.models.autoencoder import AutoencoderDetector

LOGGER = logging.getLogger(__name__)


def run_nbaiot_autoencoder_global_stage_3(config_path: str | Path) -> dict[str, Any]:
    """Ejecuta el Stage 3 del autoencoder global sobre N-BaIoT.

    Esta etapa no reentrena el modelo ni calcula métricas agregadas.
    Su responsabilidad es generar las predicciones completas sobre el conjunto
    de test global: test_benign + test_attack.

    Convención:
    - y_true = 0 para muestra benigna.
    - y_true = 1 para muestra maliciosa.
    - y_pred = 0 para predicción benigna.
    - y_pred = 1 para predicción maliciosa.
    """
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    _configure_logging(config)

    experiment_name = str(config["experiment"]["name"])
    seed = int(config["experiment"].get("seed", 42))

    data_dir = Path(config["paths"]["data_dir"])
    stage_2_run_dir = Path(config["paths"]["stage_2_run_dir"])
    model_dir = stage_2_run_dir / "model"

    output_dir = Path(config["outputs"]["output_dir"])
    overwrite = bool(config["outputs"].get("overwrite", False))

    _prepare_output_dir(output_dir=output_dir, overwrite=overwrite)
    shutil.copy2(config_path, output_dir / "config.yaml")

    metadata_columns = list(config["data"].get("metadata_columns", []))
    test_benign_file = str(config["data"].get("test_benign_file", "test_benign.parquet"))
    test_attack_file = str(config["data"].get("test_attack_file", "test_attack.parquet"))

    test_benign_path = data_dir / test_benign_file
    test_attack_path = data_dir / test_attack_file

    _validate_input_file(test_benign_path)
    _validate_input_file(test_attack_path)
    _validate_input_file(model_dir / "metadata.json")
    _validate_input_file(model_dir / "model.keras")

    threshold = float(config["threshold"]["value"])
    threshold_criterion = str(config["threshold"].get("criterion", "unknown"))
    batch_size = int(config.get("runtime", {}).get("batch_size", 100_000))

    feature_columns = _get_feature_columns(test_benign_path, metadata_columns)

    LOGGER.info("Iniciando experimento: %s", experiment_name)
    LOGGER.info("Directorio de datos: %s", data_dir)
    LOGGER.info("Directorio del Stage 2: %s", stage_2_run_dir)
    LOGGER.info("Directorio de salida: %s", output_dir)
    LOGGER.info("Threshold: %.6f (%s)", threshold, threshold_criterion)
    LOGGER.info("Número de features: %d", len(feature_columns))

    LOGGER.info("Cargando modelo global")
    model = _load_model(model_dir)

    test_predictions_path = output_dir / "test_predictions.parquet"

    LOGGER.info("Generando predicciones sobre test_benign + test_attack")
    inference_summary = _score_test_splits_to_parquet(
        model=model,
        benign_path=test_benign_path,
        attack_path=test_attack_path,
        feature_columns=feature_columns,
        threshold=threshold,
        output_path=test_predictions_path,
        batch_size=batch_size,
    )

    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "dataset": config["experiment"].get("dataset"),
        "model": config["experiment"].get("model"),
        "strategy": config["experiment"].get("strategy"),
        "seed": seed,
        "stage": 3,
        "n_features": len(feature_columns),
        "threshold": threshold,
        "threshold_criterion": threshold_criterion,
        "model_dir": str(model_dir),
        "test_predictions_path": str(test_predictions_path),
        "n_test_benign": _count_parquet_rows(test_benign_path),
        "n_test_attack": _count_parquet_rows(test_attack_path),
        "model_size_mb": _get_directory_size_mb(model_dir),
        **inference_summary,
    }

    _save_json(output_dir / "summary.json", summary)
    _save_json(output_dir / "global_summary.json", summary)

    del model
    gc.collect()

    LOGGER.info("Experimento finalizado correctamente")
    return summary


def _score_test_splits_to_parquet(
    model: AutoencoderDetector,
    benign_path: Path,
    attack_path: Path,
    feature_columns: list[str],
    threshold: float,
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("reconstruction_error", pa.float32()),
            ("y_true", pa.int8()),
            ("y_pred", pa.int8()),
            ("device_id", pa.string()),
            ("is_attack", pa.bool_()),
            ("attack_family", pa.string()),
            ("attack_type", pa.string()),
        ]
    )

    writer: pq.ParquetWriter | None = None
    total_scored_samples = 0
    total_inference_time_seconds = 0.0

    try:
        writer = pq.ParquetWriter(output_path, schema=schema)

        benign_summary = _score_split_batches_to_writer(
            model=model,
            parquet_path=benign_path,
            feature_columns=feature_columns,
            y_true=0,
            threshold=threshold,
            writer=writer,
            batch_size=batch_size,
        )

        attack_summary = _score_split_batches_to_writer(
            model=model,
            parquet_path=attack_path,
            feature_columns=feature_columns,
            y_true=1,
            threshold=threshold,
            writer=writer,
            batch_size=batch_size,
        )

        total_scored_samples = (
            benign_summary["n_scored_samples"]
            + attack_summary["n_scored_samples"]
        )
        total_inference_time_seconds = (
            benign_summary["inference_time_seconds"]
            + attack_summary["inference_time_seconds"]
        )

    finally:
        if writer is not None:
            writer.close()

    throughput = (
        total_scored_samples / total_inference_time_seconds
        if total_inference_time_seconds > 0
        else None
    )

    return {
        "n_scored_samples": int(total_scored_samples),
        "inference_time_seconds": round(float(total_inference_time_seconds), 6),
        "throughput_samples_per_second": round(float(throughput), 4)
        if throughput is not None
        else None,
    }


def _score_split_batches_to_writer(
    model: AutoencoderDetector,
    parquet_path: Path,
    feature_columns: list[str],
    y_true: int,
    threshold: float,
    writer: pq.ParquetWriter,
    batch_size: int,
) -> dict[str, Any]:
    required_metadata_columns = [
        "device_id",
        "is_attack",
        "attack_family",
        "attack_type",
    ]
    columns_to_read = feature_columns + required_metadata_columns

    parquet_file = pq.ParquetFile(parquet_path)
    n_scored_samples = 0
    inference_time_seconds = 0.0

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=columns_to_read,
    ):
        batch_df = record_batch.to_pandas()

        X = batch_df[feature_columns].to_numpy(dtype=np.float32, copy=True)

        start_time = time.perf_counter()
        reconstruction_errors = model.score_samples(X).astype(np.float32, copy=False)
        inference_time_seconds += time.perf_counter() - start_time

        y_pred = (reconstruction_errors > threshold).astype(np.int8, copy=False)
        y_true_array = np.full(
            shape=reconstruction_errors.shape,
            fill_value=y_true,
            dtype=np.int8,
        )

        table = pa.Table.from_pydict(
            {
                "reconstruction_error": reconstruction_errors,
                "y_true": y_true_array,
                "y_pred": y_pred,
                "device_id": batch_df["device_id"].astype("string").to_numpy(),
                "is_attack": batch_df["is_attack"].astype(bool).to_numpy(),
                "attack_family": batch_df["attack_family"].astype("string").to_numpy(),
                "attack_type": batch_df["attack_type"].astype("string").to_numpy(),
            },
            schema=writer.schema,
        )

        writer.write_table(table)
        n_scored_samples += int(reconstruction_errors.shape[0])

        del record_batch, batch_df, X, reconstruction_errors
        del y_pred, y_true_array, table
        gc.collect()

    return {
        "n_scored_samples": n_scored_samples,
        "inference_time_seconds": inference_time_seconds,
    }


def _load_model(path: Path) -> AutoencoderDetector:
    model = AutoencoderDetector.load(path)

    if not hasattr(model, "score_samples"):
        raise TypeError(f"El objeto cargado desde {path} no implementa score_samples().")

    return model


def _get_feature_columns(
    parquet_path: Path,
    metadata_columns: list[str],
) -> list[str]:
    schema = pq.read_schema(parquet_path)
    columns = list(schema.names)

    metadata = set(metadata_columns)
    feature_columns = [col for col in columns if col not in metadata]

    if not feature_columns:
        raise ValueError(f"No se han encontrado columnas de features en {parquet_path}.")

    return feature_columns


def _count_parquet_rows(path: Path) -> int:
    parquet_file = pq.ParquetFile(path)
    return int(parquet_file.metadata.num_rows)


def _get_directory_size_mb(path: Path) -> float:
    total_size = 0

    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    return round(total_size / (1024 * 1024), 4)


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"El directorio de salida ya existe: {output_dir}. "
                "Establece outputs.overwrite=true en la configuración para sobrescribirlo."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se ha encontrado el archivo de entrada: {path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuración YAML no válida: {path}")

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