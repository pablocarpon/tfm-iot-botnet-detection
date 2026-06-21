from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf


def parquet_batch_generator(
    parquet_file: str | Path,
    feature_columns: list[str],
    *,
    batch_size: int = 4096,
) -> Generator[np.ndarray, None, None]:
    """Lee un Parquet por batches sin cargarlo completo en RAM."""
    parquet_file = Path(parquet_file)
    parquet_reader = pq.ParquetFile(parquet_file)

    for batch in parquet_reader.iter_batches(
        batch_size=batch_size,
        columns=feature_columns,
    ):
        yield batch.to_pandas().to_numpy(dtype=np.float32)


def make_tf_dataset_from_parquet(
    parquet_file: str | Path,
    feature_columns: list[str],
    *,
    batch_size: int = 4096,
    shuffle: bool = False,
    shuffle_buffer_size: int = 50_000,
) -> tf.data.Dataset:
    """Crea un tf.data.Dataset eficiente desde un archivo Parquet."""

    def generator():
        yield from parquet_batch_generator(
            parquet_file=parquet_file,
            feature_columns=feature_columns,
            batch_size=batch_size,
        )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(
            shape=(None, len(feature_columns)),
            dtype=tf.float32,
        ),
    )

    if shuffle:
        dataset = dataset.unbatch()
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def make_autoencoder_dataset_from_parquet(
    parquet_file: str | Path,
    feature_columns: list[str],
    *,
    batch_size: int = 4096,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Crea un dataset (X, X) para entrenar autoencoders."""

    dataset = make_tf_dataset_from_parquet(
        parquet_file=parquet_file,
        feature_columns=feature_columns,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataset.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)