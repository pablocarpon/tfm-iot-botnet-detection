from __future__ import annotations

import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
import yaml

from iotbotnet.models.factory import build_model

LOGGER = logging.getLogger(__name__)


def run_nbaiot_lof_per_device_stage_1(config_path: str | Path) -> dict[str, Any]:
    """Ejecuta la etapa 1 del experimento LOF por dispositivo sobre N-BaIoT.

    Para cada dispositivo:
    1. Optimiza hiperparámetros de LOF con Optuna/TPE usando validación benigna.
    2. Entrena el modelo final únicamente con train.
    3. Guarda modelo final, trials de Optuna y distribuciones de scores train/val.

    Convención del proyecto:
    - score alto = muestra más anómala.
    - score bajo = muestra más normal.
    """
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    _configure_logging(config)

    experiment_name = str(config["experiment"]["name"])
    seed = int(config["experiment"].get("seed", 42))

    data_dir = Path(config["paths"]["data_dir"])
    output_dir = Path(config["outputs"]["output_dir"])
    overwrite = bool(config["outputs"].get("overwrite", False))

    _prepare_output_dir(output_dir=output_dir, overwrite=overwrite)
    shutil.copy2(config_path, output_dir / "config.yaml")

    devices = list(config["devices"])
    metadata_columns = list(config["data"].get("metadata_columns", []))
    train_file = config["data"].get("train_file", "train.parquet")
    val_file = config["data"].get("val_file", "val.parquet")

    n_trials = int(config["optimization"]["n_trials"])
    objective_quantile = float(config["optimization"].get("objective_quantile", 0.99))

    base_model_config = dict(config["model"])
    model_type = base_model_config.get("type")

    global_summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "dataset": config["experiment"].get("dataset"),
        "model": model_type,
        "strategy": config["experiment"].get("strategy"),
        "seed": seed,
        "n_devices": len(devices),
        "objective": f"minimize_p{int(objective_quantile * 100)}_validation_score_per_device",
        "devices": {},
    }

    LOGGER.info("Starting experiment: %s", experiment_name)
    LOGGER.info("Data directory: %s", data_dir)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Model type: %s", model_type)

    for device in devices:
        LOGGER.info("[%s] Starting optimization", device)

        device_dir = data_dir / device
        train_path = device_dir / train_file
        val_path = device_dir / val_file

        _validate_input_file(train_path)
        _validate_input_file(val_path)

        device_output_dir = output_dir / "devices" / device
        device_output_dir.mkdir(parents=True, exist_ok=True)

        feature_columns = _get_feature_columns(train_path, metadata_columns)

        study = _create_study(
            experiment_name=experiment_name,
            device=device,
            seed=seed,
        )

        objective = _build_objective(
            train_path=train_path,
            val_path=val_path,
            feature_columns=feature_columns,
            config=config,
            base_model_config=base_model_config,
            objective_quantile=objective_quantile,
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = dict(study.best_params)
        best_value = float(study.best_value)

        final_model_config = _build_model_config(
            base_model_config=base_model_config,
            trial_params=best_params,
        )

        LOGGER.info("[%s] Best objective value: %.6f", device, best_value)
        LOGGER.info("[%s] Best model config: %s", device, final_model_config)

        _save_json(device_output_dir / "best_params.json", final_model_config)
        study.trials_dataframe().to_csv(
            device_output_dir / "optuna_trials.csv",
            index=False,
        )

        LOGGER.info("[%s] Training final model", device)
        final_model = _train_model(
            parquet_path=train_path,
            feature_columns=feature_columns,
            model_config=final_model_config,
        )
        final_model.save(device_output_dir / "model.joblib")

        LOGGER.info("[%s] Scoring train split", device)
        train_scores = _score_file(
            model=final_model,
            parquet_path=train_path,
            feature_columns=feature_columns,
        )
        _save_scores(train_scores, device_output_dir / "train_scores.parquet")
        del train_scores
        gc.collect()

        LOGGER.info("[%s] Scoring validation split", device)
        val_scores = _score_file(
            model=final_model,
            parquet_path=val_path,
            feature_columns=feature_columns,
        )
        _save_scores(val_scores, device_output_dir / "val_scores.parquet")
        del val_scores, final_model
        gc.collect()

        global_summary["devices"][device] = {
            "best_value": best_value,
            "best_params": final_model_config,
            "n_trials": n_trials,
            "n_features": len(feature_columns),
        }

        _save_json(output_dir / "global_summary.json", global_summary)
        LOGGER.info("[%s] Finished", device)

    LOGGER.info("Experiment finished successfully")
    return global_summary


def _build_objective(
    train_path: Path,
    val_path: Path,
    feature_columns: list[str],
    config: dict[str, Any],
    base_model_config: dict[str, Any],
    objective_quantile: float,
):
    search_space = config["search_space"]

    n_neighbors_low = int(search_space["n_neighbors"]["low"])
    n_neighbors_high = int(search_space["n_neighbors"]["high"])
    metric_choices = list(search_space["metric"]["choices"])

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "n_neighbors": trial.suggest_int(
                "n_neighbors",
                n_neighbors_low,
                n_neighbors_high,
            ),
            "metric": trial.suggest_categorical("metric", metric_choices),
        }

        model_config = _build_model_config(
            base_model_config=base_model_config,
            trial_params=trial_params,
        )

        model = _train_model(
            parquet_path=train_path,
            feature_columns=feature_columns,
            model_config=model_config,
        )

        val_scores = _score_file(
            model=model,
            parquet_path=val_path,
            feature_columns=feature_columns,
        )

        objective_value = float(np.quantile(val_scores, objective_quantile))

        del model, val_scores
        gc.collect()

        return objective_value

    return objective


def _build_model_config(
    base_model_config: dict[str, Any],
    trial_params: dict[str, Any],
) -> dict[str, Any]:
    """Combina la configuración fija del modelo con los parámetros de Optuna."""
    model_config = dict(base_model_config)

    fixed_params = model_config.pop("fixed_params", {})
    model_config.update(fixed_params)
    model_config.update(trial_params)

    return model_config


def _train_model(
    parquet_path: Path,
    feature_columns: list[str],
    model_config: dict[str, Any],
) -> Any:
    X = _load_features_as_numpy(parquet_path, feature_columns)

    model = build_model(model_config)
    model.fit(X)

    del X
    gc.collect()

    return model


def _score_file(
    model: Any,
    parquet_path: Path,
    feature_columns: list[str],
) -> np.ndarray:
    X = _load_features_as_numpy(parquet_path, feature_columns)

    scores = model.score_samples(X).astype(np.float32, copy=False)

    del X
    gc.collect()

    return scores


def _load_features_as_numpy(
    parquet_path: Path,
    feature_columns: list[str],
) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=feature_columns)

    X = df.to_numpy(dtype=np.float32, copy=True)

    del df
    gc.collect()

    return X


def _save_scores(scores: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame({"score": scores})
    scores_df.to_parquet(output_path, index=False)

    del scores_df
    gc.collect()


def _get_feature_columns(
    parquet_path: Path,
    metadata_columns: list[str],
) -> list[str]:
    schema = pq.read_schema(parquet_path)
    columns = list(schema.names)

    metadata = set(metadata_columns)
    feature_columns = [col for col in columns if col not in metadata]

    if not feature_columns:
        raise ValueError(f"No feature columns found in {parquet_path}.")

    return feature_columns


def _create_study(
    experiment_name: str,
    device: str,
    seed: int,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)

    return optuna.create_study(
        study_name=f"{experiment_name}_{device}",
        direction="minimize",
        sampler=sampler,
    )


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