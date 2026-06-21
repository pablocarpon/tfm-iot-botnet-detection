from __future__ import annotations

import gc
from pathlib import Path

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = ["is_attack", "attack_family", "attack_type", "device_id"]


def get_feature_columns(parquet_file: str | Path) -> list[str]:
    """Devuelve únicamente las columnas numéricas usadas por el modelo."""
    parquet_file = pq.ParquetFile(parquet_file)
    columns = parquet_file.schema_arrow.names

    return [col for col in columns if col not in NON_FEATURE_COLUMNS]


def fit_standard_scaler(
    train_file: str | Path,
    feature_columns: list[str] | None = None,
) -> StandardScaler:
    """Ajusta el scaler usando el conjunto de entrenamiento benigno completo."""
    train_file = Path(train_file)

    if feature_columns is None:
        feature_columns = get_feature_columns(train_file)

    df = pd.read_parquet(
        train_file,
        columns=feature_columns,
        engine="pyarrow",
    )

    scaler = StandardScaler()
    scaler.fit(df.to_numpy(dtype="float32"))

    del df
    gc.collect()

    return scaler


def transform_parquet_file(
    input_file: str | Path,
    output_file: str | Path,
    scaler: StandardScaler,
    feature_columns: list[str],
    *,
    batch_size: int = 100_000,
    compression: str = "snappy",
) -> None:
    """
    Escala un archivo Parquet por batches.

    Esta función evita cargar todo el archivo en RAM, lo cual es importante
    especialmente para test_attack.parquet en los splits globales.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        output_file.unlink()

    parquet_file = pq.ParquetFile(input_file)
    writer: pq.ParquetWriter | None = None

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            df[feature_columns] = scaler.transform(
                df[feature_columns].to_numpy(dtype="float32")
            ).astype("float32")

            table = pa.Table.from_pandas(
                df,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression=compression,
                )

            writer.write_table(table)

            del df, table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    gc.collect()


def scale_split_directory(
    input_folder: str | Path,
    output_folder: str | Path,
    scaler_output_file: str | Path,
    *,
    batch_size: int = 100_000,
    compression: str = "snappy",
) -> None:
    """
    Escala un directorio de splits.

    Espera:
    - train.parquet
    - val.parquet
    - test_benign.parquet
    - test_attack.parquet
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    scaler_output_file = Path(scaler_output_file)

    output_folder.mkdir(parents=True, exist_ok=True)
    scaler_output_file.parent.mkdir(parents=True, exist_ok=True)

    train_file = input_folder / "train.parquet"
    feature_columns = get_feature_columns(train_file)

    scaler = fit_standard_scaler(
        train_file=train_file,
        feature_columns=feature_columns,
    )

    joblib.dump(scaler, scaler_output_file)

    for file_name in [
        "train.parquet",
        "val.parquet",
        "test_benign.parquet",
        "test_attack.parquet",
    ]:
        transform_parquet_file(
            input_file=input_folder / file_name,
            output_file=output_folder / file_name,
            scaler=scaler,
            feature_columns=feature_columns,
            batch_size=batch_size,
            compression=compression,
        )

    del scaler
    gc.collect()