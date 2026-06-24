from __future__ import annotations

from pathlib import Path
from tqdm.auto import tqdm

from iotbotnet.data.loaders import (
    create_nbaiot_device_parquets,
    load_nbaiot_csv_file,
)
from iotbotnet.data.preprocessing import scale_split_directory
from iotbotnet.data.splits import create_global_splits, create_per_device_splits


N_BAIOT_REPOSITORY_PATH = "mkashifn/nbaiot-dataset"
N_BAIOT_OUTPUT_ROOT = "data/processed/n_baiot"
N_BAIOT_DATA_SUMMARY_FILE = "data_summary.csv"


def get_nbaiot_csv_file_names(repository_path: str) -> list[str]:
    """
    Obtiene los nombres de los archivos CSV de datos desde data_summary.csv.

    No guarda el archivo de metadatos en disco. Solo lo usa para evitar
    hardcodear la lista de archivos del dataset.
    """
    data_summary_df = load_nbaiot_csv_file(
        repository_path=repository_path,
        file_name=N_BAIOT_DATA_SUMMARY_FILE,
    )

    data_summary_df.columns = data_summary_df.columns.str.strip()

    if "File Name" not in data_summary_df.columns:
        raise KeyError(
            "Column 'File Name' was not found in data_summary.csv. "
            f"Available columns: {data_summary_df.columns.tolist()}"
        )

    csv_file_names = (
        data_summary_df["File Name"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    return csv_file_names


def prepare_n_baiot_data(
    output_root: str | Path = N_BAIOT_OUTPUT_ROOT,
    *,
    train_size: float = 0.6,
    val_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
    compression: str = "snappy",
    batch_size: int = 100_000,
) -> None:
    """
    Ejecuta el pipeline completo de preparación del dataset N-BaIoT.

    El pipeline genera:
    - Archivos Parquet base por dispositivo.
    - Splits por dispositivo.
    - Splits globales.
    - Versiones estandarizadas de todos los splits.
    - Scalers ajustados únicamente sobre train.
    """
    output_root = Path(output_root)

    base_folder = output_root / "base"

    per_device_splits_folder = output_root / "splits" / "per_device"
    global_splits_folder = output_root / "splits" / "global"

    scaled_per_device_folder = output_root / "scaled" / "per_device"
    scaled_global_folder = output_root / "scaled" / "global"

    scalers_folder = Path("outputs/models/n_baiot/scalers")

    csv_file_names = get_nbaiot_csv_file_names(
        repository_path=N_BAIOT_REPOSITORY_PATH,
    )

    print("[1/5] Creating base Parquet files...")
    create_nbaiot_device_parquets(
        repository_path=N_BAIOT_REPOSITORY_PATH,
        input_csv_file_names=csv_file_names,
        output_folder=base_folder,
        compression=compression,
    )

    print("[2/5] Creating per-device splits...")
    create_per_device_splits(
        base_folder=base_folder,
        output_folder=per_device_splits_folder,
        train_size=train_size,
        val_size=val_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    print("[3/5] Creating global splits...")
    create_global_splits(
        base_folder=base_folder,
        output_folder=global_splits_folder,
        train_size=train_size,
        val_size=val_size,
        shuffle=shuffle,
        random_state=random_state,
        compression=compression,
    )

    print("[4/5] Scaling per-device splits...")
    for device_folder in tqdm(sorted(per_device_splits_folder.glob("device_*")), desc="Scaling per-device splits"):
        scale_split_directory(
            input_folder=device_folder,
            output_folder=scaled_per_device_folder / device_folder.name,
            scaler_output_file=(
                scalers_folder
                / "per_device"
                / f"{device_folder.name}_standard_scaler.joblib"
            ),
            batch_size=batch_size,
            compression=compression,
        )

    print("[5/5] Scaling global splits...")
    scale_split_directory(
        input_folder=global_splits_folder,
        output_folder=scaled_global_folder,
        scaler_output_file=scalers_folder / "global" / "standard_scaler.joblib",
        batch_size=batch_size,
        compression=compression,
    )