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
import optuna
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


def run_nbaiot_autoencoder_global_stage_2(
    config_path: str | Path,
) -> dict[str, Any]:
    """Ejecuta el Stage 2 del autoencoder global sobre N-BaIoT.

    Esta etapa:
    1. Optimiza hiperparámetros con Optuna/TPE usando train benigno global y
       validación benigna global.
    2. Minimiza el percentil configurado de los errores de reconstrucción sobre
       validación benigna.
    3. Reentrena desde cero el mejor modelo encontrado usando la configuración
       de entrenamiento final.
    4. Guarda modelo final, configuración, histórico de entrenamiento, trials de
       Optuna y distribuciones de errores de reconstrucción train/val.

    Esta etapa NO selecciona el threshold final y NO evalúa sobre test.
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

    metadata_columns = list(config["data"].get("metadata_columns", []))
    train_file = str(config["data"].get("train_file", "train.parquet"))
    val_file = str(config["data"].get("val_file", "val.parquet"))

    train_path = data_dir / train_file
    val_path = data_dir / val_file
    _validate_input_file(train_path)
    _validate_input_file(val_path)

    n_trials = int(config["optimization"]["n_trials"])
    objective_quantile = float(config["optimization"].get("objective_quantile", 0.99))
    _validate_quantile(objective_quantile)

    optimization_training_config = dict(config["training"])
    final_training_config = _build_final_training_config(config)

    opt_batch_size = int(optimization_training_config.get("batch_size", 1024))
    opt_epochs = int(optimization_training_config.get("epochs", 100))
    opt_verbose = int(optimization_training_config.get("verbose", 1))
    opt_shuffle_train = bool(optimization_training_config.get("shuffle_train", False))
    opt_shuffle_buffer_size = int(
        optimization_training_config.get("shuffle_buffer_size", 50_000)
    )

    final_batch_size = int(final_training_config.get("batch_size", opt_batch_size))
    final_epochs = int(final_training_config.get("epochs", opt_epochs))
    final_verbose = int(final_training_config.get("verbose", opt_verbose))
    final_shuffle_train = bool(final_training_config.get("shuffle_train", opt_shuffle_train))
    final_shuffle_buffer_size = int(
        final_training_config.get("shuffle_buffer_size", opt_shuffle_buffer_size)
    )

    base_model_config = dict(config["model"])
    model_type = str(base_model_config.get("type"))
    feature_columns = _get_feature_columns(train_path, metadata_columns)

    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "dataset": config["experiment"].get("dataset"),
        "model": model_type,
        "strategy": config["experiment"].get("strategy"),
        "seed": seed,
        "stage": 2,
        "n_features": len(feature_columns),
        "n_trials": n_trials,
        "objective": (
            "minimize_"
            f"p{int(objective_quantile * 100)}_validation_reconstruction_error"
        ),
        "objective_quantile": objective_quantile,
        "optimization_training": {
            "epochs": opt_epochs,
            "batch_size": opt_batch_size,
            "shuffle_train": opt_shuffle_train,
            "early_stopping": optimization_training_config.get("early_stopping", {}),
            "reduce_lr_on_plateau": optimization_training_config.get(
                "reduce_lr_on_plateau", {}
            ),
        },
        "final_training": {
            "epochs": final_epochs,
            "batch_size": final_batch_size,
            "shuffle_train": final_shuffle_train,
            "early_stopping": final_training_config.get("early_stopping", {}),
            "reduce_lr_on_plateau": final_training_config.get("reduce_lr_on_plateau", {}),
        },
    }

    LOGGER.info("Starting experiment: %s", experiment_name)
    LOGGER.info("Data directory: %s", data_dir)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Model type: %s", model_type)
    LOGGER.info("Trials: %d", n_trials)
    LOGGER.info("Features: %d", len(feature_columns))

    study = _create_study(experiment_name=experiment_name, seed=seed)
    objective = _build_objective(
        train_path=train_path,
        val_path=val_path,
        feature_columns=feature_columns,
        config=config,
        base_model_config=base_model_config,
        training_config=optimization_training_config,
        input_dim=len(feature_columns),
        objective_quantile=objective_quantile,
        batch_size=opt_batch_size,
        epochs=opt_epochs,
        verbose=opt_verbose,
        shuffle_train=opt_shuffle_train,
        shuffle_buffer_size=opt_shuffle_buffer_size,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_value = float(study.best_value)
    final_model_config = _build_model_config(
        base_model_config=base_model_config,
        trial_params=best_params,
        input_dim=len(feature_columns),
    )

    LOGGER.info("Best objective value: %.8f", best_value)
    LOGGER.info("Best model config: %s", final_model_config)

    _save_json(output_dir / "best_params.json", final_model_config)
    study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False)

    LOGGER.info("Training final model")
    final_model, history_df = _train_model(
        train_path=train_path,
        val_path=val_path,
        feature_columns=feature_columns,
        model_config=final_model_config,
        training_config=final_training_config,
        batch_size=final_batch_size,
        epochs=final_epochs,
        verbose=final_verbose,
        shuffle_train=final_shuffle_train,
        shuffle_buffer_size=final_shuffle_buffer_size,
    )

    _save_json(output_dir / "model_config.json", final_model_config)
    history_df.to_csv(output_dir / "history.csv", index=False)
    final_model.save(output_dir / "model")

    train_score_dataset = make_tf_dataset_from_parquet(
        parquet_file=train_path,
        feature_columns=feature_columns,
        batch_size=final_batch_size,
        shuffle=False,
    )
    val_score_dataset = make_tf_dataset_from_parquet(
        parquet_file=val_path,
        feature_columns=feature_columns,
        batch_size=final_batch_size,
        shuffle=False,
    )

    LOGGER.info("Scoring train split")
    train_scores = final_model.score_samples(train_score_dataset)
    _save_scores(train_scores, output_dir / "train_reconstruction_errors.parquet")

    LOGGER.info("Scoring validation split")
    val_scores = final_model.score_samples(val_score_dataset)
    _save_scores(val_scores, output_dir / "val_reconstruction_errors.parquet")

    best_epoch, best_val_loss = _get_best_epoch_and_val_loss(history_df)
    final_val_objective = float(np.quantile(val_scores, objective_quantile))

    summary.update(
        {
            "best_objective_value": best_value,
            "final_model_validation_objective": final_val_objective,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_params": final_model_config,
            "train_error_summary": _summarize_scores(train_scores),
            "val_error_summary": _summarize_scores(val_scores),
        }
    )

    _save_json(output_dir / "summary.json", summary)
    _save_json(output_dir / "global_summary.json", summary)

    del (
        final_model,
        history_df,
        train_score_dataset,
        val_score_dataset,
        train_scores,
        val_scores,
    )
    tf.keras.backend.clear_session()
    gc.collect()

    LOGGER.info("Experiment finished successfully")
    return summary


def _build_objective(
    *,
    train_path: Path,
    val_path: Path,
    feature_columns: list[str],
    config: dict[str, Any],
    base_model_config: dict[str, Any],
    training_config: dict[str, Any],
    input_dim: int,
    objective_quantile: float,
    batch_size: int,
    epochs: int,
    verbose: int,
    shuffle_train: bool,
    shuffle_buffer_size: int,
):
    search_space = dict(config["search_space"])

    def objective(trial: optuna.Trial) -> float:
        trial_params = _suggest_trial_params(trial, search_space)
        model_config = _build_model_config(
            base_model_config=base_model_config,
            trial_params=trial_params,
            input_dim=input_dim,
        )

        model, history_df = _train_model(
            train_path=train_path,
            val_path=val_path,
            feature_columns=feature_columns,
            model_config=model_config,
            training_config=training_config,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle_train=shuffle_train,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        val_score_dataset = make_tf_dataset_from_parquet(
            parquet_file=val_path,
            feature_columns=feature_columns,
            batch_size=batch_size,
            shuffle=False,
        )
        val_scores = model.score_samples(val_score_dataset)
        objective_value = float(np.quantile(val_scores, objective_quantile))

        best_epoch, best_val_loss = _get_best_epoch_and_val_loss(history_df)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_val_loss", best_val_loss)
        trial.set_user_attr("epochs_trained", int(len(history_df)))
        trial.set_user_attr("val_error_mean", float(np.mean(val_scores)))
        trial.set_user_attr("val_error_std", float(np.std(val_scores)))
        trial.set_user_attr("val_error_p95", float(np.quantile(val_scores, 0.95)))
        trial.set_user_attr("val_error_p99", float(np.quantile(val_scores, 0.99)))
        trial.set_user_attr("val_error_p99_5", float(np.quantile(val_scores, 0.995)))
        trial.set_user_attr(
            f"val_error_p{int(objective_quantile * 100)}",
            objective_value,
        )

        del model, history_df, val_score_dataset, val_scores
        tf.keras.backend.clear_session()
        gc.collect()

        return objective_value

    return objective


def _suggest_trial_params(
    trial: optuna.Trial,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    for name, spec in search_space.items():
        spec = dict(spec)

        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
            continue

        if "low" not in spec or "high" not in spec:
            raise ValueError(
                f"Invalid search space for '{name}'. Expected 'choices' or 'low'/'high'."
            )

        low = spec["low"]
        high = spec["high"]
        log = bool(spec.get("log", False))

        if isinstance(low, int) and isinstance(high, int) and not spec.get("float", False):
            params[name] = trial.suggest_int(name, int(low), int(high), log=log)
        else:
            if log and float(low) <= 0:
                raise ValueError(
                    f"Invalid log search space for '{name}'. Low value must be > 0."
                )
            params[name] = trial.suggest_float(name, float(low), float(high), log=log)

    return params


def _build_model_config(
    *,
    base_model_config: dict[str, Any],
    trial_params: dict[str, Any],
    input_dim: int,
) -> dict[str, Any]:
    """Combina parámetros fijos, parámetros sugeridos por Optuna e input_dim."""
    model_config = dict(base_model_config)

    fixed_params = dict(model_config.pop("fixed_params", {}))
    model_config.update(fixed_params)
    model_config.update(trial_params)
    model_config["input_dim"] = input_dim

    if "reconstruction_error" not in model_config:
        model_config["reconstruction_error"] = model_config.get("loss", "mse")

    return model_config


def _train_model(
    *,
    train_path: Path,
    val_path: Path,
    feature_columns: list[str],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    batch_size: int,
    epochs: int,
    verbose: int,
    shuffle_train: bool,
    shuffle_buffer_size: int,
) -> tuple[Any, pd.DataFrame]:
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

    model = build_model(model_config)
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

    del train_dataset, val_dataset
    gc.collect()

    return model, history_df


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

    reduce_lr_config = dict(training_config.get("reduce_lr_on_plateau", {}))
    if reduce_lr_config.get("enabled", False):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=str(reduce_lr_config.get("monitor", "val_loss")),
                factor=float(reduce_lr_config.get("factor", 0.5)),
                patience=int(reduce_lr_config.get("patience", 5)),
                min_delta=float(reduce_lr_config.get("min_delta", 1e-5)),
                min_lr=float(reduce_lr_config.get("min_lr", 1e-6)),
                mode=str(reduce_lr_config.get("mode", "min")),
                verbose=int(reduce_lr_config.get("verbose", 0)),
            )
        )

    return callbacks


def _build_final_training_config(config: dict[str, Any]) -> dict[str, Any]:
    final_training_config = dict(config.get("training", {}))
    final_training_config.update(dict(config.get("final_training", {})))
    return final_training_config


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
        "p99_5": float(np.quantile(scores, 0.995)),
        "max": float(np.max(scores)),
    }


def _save_scores(scores: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame(
        {"reconstruction_error": scores.astype(np.float32, copy=False)}
    )
    scores_df.to_parquet(output_path, index=False)

    del scores_df
    gc.collect()


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


def _create_study(
    *,
    experiment_name: str,
    seed: int,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)

    return optuna.create_study(
        study_name=experiment_name,
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


def _validate_quantile(quantile: float) -> None:
    if not 0 < quantile < 1:
        raise ValueError(f"objective_quantile must be between 0 and 1, got {quantile}.")


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