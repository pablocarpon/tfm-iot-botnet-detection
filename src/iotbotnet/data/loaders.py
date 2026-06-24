"""
Loaders del dataset N-BaIoT.

Formato de los archivos CSV en crudo:
    {device_id}.benign.csv
    {device_id}.{attack_family}.{attack_type}.csv

Salida de los loaders:
    Un archivo Parquet por cada dispositivo:
        device_1.parquet
        ...
        device_9.parquet
"""

from __future__ import annotations

import gc
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def parse_nbaiot_filename(file_name: str | Path) -> dict:
    file_name = Path(file_name)
    parts = file_name.name.replace(".csv", "").split(".")

    if len(parts) == 2 and parts[1] == "benign":
        return {
            "device_id": int(parts[0]),
            "is_attack": False,
            "attack_family": "benign",
            "attack_type": "benign",
        }

    if len(parts) == 3:
        return {
            "device_id": int(parts[0]),
            "is_attack": True,
            "attack_family": parts[1],
            "attack_type": parts[2],
        }

    raise ValueError(f"Unexpected N-BaIoT filename format: {file_name.name}")


def load_nbaiot_csv_file(
    repository_path: str,
    file_name: str,
) -> pd.DataFrame:
    """
    Carga un archivo CSV individual del dataset N-BaIoT desde Kaggle.

    Se asume que `dataset_load` devuelve un DataFrame correspondiente
    únicamente al archivo indicado mediante `path=file_name`.
    """
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        repository_path,
        path=file_name,
    )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"kagglehub.dataset_load did not return a pandas DataFrame for file: {file_name}"
        )

    return df


def create_nbaiot_device_parquets(
    repository_path: str,
    input_csv_file_names: list[str],
    output_folder: str | Path,
    *,
    compression: str = "snappy",
) -> None:
    """
    Convierte los archivos CSV del dataset N-BaIoT de Kaggle en archivos Parquet, uno por cada dispositivo.

    La función está optimizada para no sobrecargar la memoria RAM:
    - Carga un único CSV cada vez
    - Escribe incrementalmente cada CSV en el Parquet correspondiente al dispositivo
    - Evita concatenar DataFrames en memoria
    - Libera explícitamente objetos pesados tras procesar cada archivo
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not isinstance(input_csv_file_names, list) or len(input_csv_file_names) == 0:
        raise ValueError("input_csv_file_names must be a non-empty list of strings")

    writers: dict[int, pq.ParquetWriter] = {}
    initialized_devices: set[int] = set()

    try:
        for file_name in tqdm(input_csv_file_names, desc="Creating base Parquet files"):
            metadata = parse_nbaiot_filename(file_name)
            device_id = metadata["device_id"]

            output_path = output_folder / f"device_{device_id}.parquet"

            if device_id not in initialized_devices:
                if output_path.exists():
                    output_path.unlink()
                initialized_devices.add(device_id)

            df = load_nbaiot_csv_file(repository_path, file_name)

            df = df.astype("float32", copy=False)

            metadata_df = pd.DataFrame(
                {
                    "is_attack": metadata["is_attack"],
                    "attack_family": metadata["attack_family"],
                    "attack_type": metadata["attack_type"],
                },
                index=df.index,
            )

            df = pd.concat([df, metadata_df], axis=1)

            table = pa.Table.from_pandas(
                df,
                preserve_index=False,
            )

            writer = writers.get(device_id)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression=compression,
                )
                writers[device_id] = writer

            writer.write_table(table)

            del df
            del table
            gc.collect()

    finally:
        for writer in writers.values():
            writer.close()

    gc.collect()