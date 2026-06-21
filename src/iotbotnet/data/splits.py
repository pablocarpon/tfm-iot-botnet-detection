from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def create_per_device_splits(
    base_folder: str | Path,
    output_folder: str | Path,
    *,
    train_size: float = 0.6,
    val_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
) -> None:
    """
    Crea splits independientes por dispositivo.

    Solo se dividen las muestras benignas:
    - train.parquet
    - val.parquet
    - test_benign.parquet

    Las muestras de ataque se guardan completas en:
    - test_attack.parquet
    """
    base_folder = Path(base_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for device_file in sorted(base_folder.glob("device_*.parquet")):
        device_name = device_file.stem
        device_output_folder = output_folder / device_name
        device_output_folder.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(device_file, engine="pyarrow")

        benign_df = df[df["is_attack"] == False]
        attack_df = df[df["is_attack"] == True]

        if shuffle:
            benign_df = benign_df.sample(
                frac=1.0,
                random_state=random_state,
            ).reset_index(drop=True)

        train_df, val_df, test_benign_df = _split_benign_dataframe(
            benign_df=benign_df,
            train_size=train_size,
            val_size=val_size,
        )

        train_df.to_parquet(device_output_folder / "train.parquet", index=False)
        val_df.to_parquet(device_output_folder / "val.parquet", index=False)
        test_benign_df.to_parquet(device_output_folder / "test_benign.parquet", index=False)
        attack_df.to_parquet(device_output_folder / "test_attack.parquet", index=False)

        del df, benign_df, attack_df, train_df, val_df, test_benign_df
        gc.collect()


def create_global_splits(
    base_folder: str | Path,
    output_folder: str | Path,
    *,
    train_size: float = 0.6,
    val_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
    compression: str = "snappy",
) -> None:
    """
    Crea splits globales sin cargar todos los dispositivos a la vez.

    Estrategia:
    - Cada dispositivo se divide individualmente en train/val/test_benign.
    - Después, cada parte se escribe incrementalmente en el split global.
    - Las muestras de ataque se guardan completas en test_attack.
    - Se añade device_id como metadato, pero no debe usarse como feature.
    """
    base_folder = Path(base_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file_paths = {
        "train": output_folder / "train.parquet",
        "val": output_folder / "val.parquet",
        "test_benign": output_folder / "test_benign.parquet",
        "test_attack": output_folder / "test_attack.parquet",
    }

    for output_file in output_file_paths.values():
        if output_file.exists():
            output_file.unlink()

    writers: dict[str, pq.ParquetWriter | None] = {
        split_name: None for split_name in output_file_paths
    }

    try:
        for device_file in sorted(base_folder.glob("device_*.parquet")):
            device_id = int(device_file.stem.split("_")[1])

            df = pd.read_parquet(device_file, engine="pyarrow")
            df["device_id"] = device_id

            benign_df = df[df["is_attack"] == False]
            attack_df = df[df["is_attack"] == True]

            if shuffle:
                benign_df = benign_df.sample(
                    frac=1.0,
                    random_state=random_state,
                ).reset_index(drop=True)

            train_df, val_df, test_benign_df = _split_benign_dataframe(
                benign_df=benign_df,
                train_size=train_size,
                val_size=val_size,
            )

            writers["train"] = _write_parquet_chunk(
                train_df,
                output_file_paths["train"],
                writers["train"],
                compression,
            )
            writers["val"] = _write_parquet_chunk(
                val_df,
                output_file_paths["val"],
                writers["val"],
                compression,
            )
            writers["test_benign"] = _write_parquet_chunk(
                test_benign_df,
                output_file_paths["test_benign"],
                writers["test_benign"],
                compression,
            )
            writers["test_attack"] = _write_parquet_chunk(
                attack_df,
                output_file_paths["test_attack"],
                writers["test_attack"],
                compression,
            )

            del df, benign_df, attack_df, train_df, val_df, test_benign_df
            gc.collect()

    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    gc.collect()


def _split_benign_dataframe(
    benign_df: pd.DataFrame,
    train_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide un DataFrame benigno en train, validation y test."""
    n = len(benign_df)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = benign_df.iloc[:train_end]
    val_df = benign_df.iloc[train_end:val_end]
    test_benign_df = benign_df.iloc[val_end:]

    return train_df, val_df, test_benign_df


def _write_parquet_chunk(
    df: pd.DataFrame,
    output_file: Path,
    writer: pq.ParquetWriter | None,
    compression: str,
) -> pq.ParquetWriter | None:
    """Escribe un fragmento en Parquet sin acumular dispositivos en memoria."""
    if df.empty:
        return writer

    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            output_file,
            table.schema,
            compression=compression,
        )

    writer.write_table(table)

    del table
    return writer