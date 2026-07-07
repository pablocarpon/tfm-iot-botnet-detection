from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


class AutoencoderDetector:
    """Detector de anomalías basado en un autoencoder denso de Keras."""

    VALID_RECONSTRUCTION_ERRORS = ("mse", "mae", "rmse")

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        output_activation: str = "linear",
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        l2_reg: float = 0.0,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        loss: str = "mse",
        reconstruction_error: str = "mse",
    ) -> None:
        if reconstruction_error not in self.VALID_RECONSTRUCTION_ERRORS:
            raise ValueError(
                f"Invalid reconstruction_error='{reconstruction_error}'. "
                f"Valid values are {sorted(self.VALID_RECONSTRUCTION_ERRORS)}."
            )

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = hidden_dims or [86, 57, 38]
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = float(dropout_rate)
        self.batch_norm = bool(batch_norm)
        self.l2_reg = float(l2_reg)
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.reconstruction_error = reconstruction_error

        self.model, self.encoder = self._build_model()
        self.model.compile(
            optimizer=self._build_optimizer(),
            loss=self.loss,
        )

    def _build_model(self) -> tuple[tf.keras.Model, tf.keras.Model]:
        inputs = tf.keras.Input(shape=(self.input_dim,), name="input")
        x = inputs

        for index, hidden_dim in enumerate(self.hidden_dims):
            x = self._dense_block(
                x,
                units=int(hidden_dim),
                name_prefix=f"encoder_{index + 1}",
            )

        latent = self._dense_block(
            x,
            units=self.latent_dim,
            name_prefix="latent",
            apply_dropout=False,
            dense_name="latent",
        )

        x = latent

        for index, hidden_dim in enumerate(reversed(self.hidden_dims)):
            x = self._dense_block(
                x,
                units=int(hidden_dim),
                name_prefix=f"decoder_{index + 1}",
            )

        outputs = tf.keras.layers.Dense(
            self.input_dim,
            activation=self.output_activation,
            kernel_regularizer=self._kernel_regularizer(),
            name="reconstruction",
        )(x)

        autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
        encoder = tf.keras.Model(inputs=inputs, outputs=latent, name="encoder")

        return autoencoder, encoder

    def _dense_block(
        self,
        x: tf.Tensor,
        *,
        units: int,
        name_prefix: str,
        apply_dropout: bool = True,
        dense_name: str | None = None,
    ) -> tf.Tensor:
        """Bloque denso configurable.

        Orden aplicado:
        Dense -> BatchNorm opcional -> activación -> Dropout opcional.
        """
        x = tf.keras.layers.Dense(
            units,
            activation=None,
            kernel_regularizer=self._kernel_regularizer(),
            name=dense_name or f"{name_prefix}_dense",
        )(x)

        if self.batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_batch_norm")(x)

        x = self._activation_layer(name=f"{name_prefix}_activation")(x)

        if apply_dropout and self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(
                self.dropout_rate,
                name=f"{name_prefix}_dropout",
            )(x)

        return x

    def _activation_layer(self, *, name: str) -> tf.keras.layers.Layer:
        activation_name = self.activation.lower()

        if activation_name in {"leaky_relu", "leakyrelu"}:
            return tf.keras.layers.LeakyReLU(negative_slope=0.01, name=name)

        return tf.keras.layers.Activation(self.activation, name=name)

    def _kernel_regularizer(self) -> tf.keras.regularizers.Regularizer | None:
        if self.l2_reg <= 0:
            return None

        return tf.keras.regularizers.l2(self.l2_reg)

    def _build_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer_name = self.optimizer.lower()

        if optimizer_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if optimizer_name == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        if optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        raise ValueError(
            f"Unsupported optimizer='{self.optimizer}'. "
            "Supported values are: adam, rmsprop, sgd."
        )

    def fit(
        self,
        data: Any,
        *,
        validation_data: Any | None = None,
        epochs: int = 50,
        batch_size: int | None = None,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> tf.keras.callbacks.History:
        """Entrena el autoencoder.

        `data` puede ser:
        - np.ndarray con shape (n_samples, n_features)
        - tf.data.Dataset que devuelva (X, X)
        """
        return self.model.fit(
            data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs,
        )

    def reconstruct(self, data: Any, *, verbose: int = 0) -> np.ndarray:
        """Reconstruye las muestras de entrada."""
        predictions = self.model.predict(data, verbose=verbose)

        if isinstance(predictions, tf.Tensor):
            predictions = predictions.numpy()

        return np.asarray(predictions)

    def encode(self, data: Any, *, verbose: int = 0) -> np.ndarray:
        """Devuelve la representación latente."""
        embeddings = self.encoder.predict(data, verbose=verbose)

        if isinstance(embeddings, tf.Tensor):
            embeddings = embeddings.numpy()

        return np.asarray(embeddings)

    def score_samples(self, data: Any) -> np.ndarray:
        """Devuelve un score de anomalía por muestra.

        Convención:
        - score alto = muestra más anómala
        """
        x_true = self._collect_inputs(data)
        x_pred = self.reconstruct(data)

        errors = x_true - x_pred

        if self.reconstruction_error == "mse":
            scores = np.mean(np.square(errors), axis=1)
        elif self.reconstruction_error == "mae":
            scores = np.mean(np.abs(errors), axis=1)
        elif self.reconstruction_error == "rmse":
            scores = np.sqrt(np.mean(np.square(errors), axis=1))
        else:
            raise RuntimeError(
                f"Unexpected reconstruction_error='{self.reconstruction_error}'."
            )

        return np.asarray(scores, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        """Guarda el modelo Keras y la metadata necesaria para recargarlo."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path / "model.keras")

        metadata = self.get_params()
        with (path / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "AutoencoderDetector":
        """Carga un detector guardado previamente."""
        path = Path(path)

        metadata_path = path / "metadata.json"
        model_path = path / "model.keras"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Keras model file not found: {model_path}")

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        detector = cls(**metadata)
        detector.model = tf.keras.models.load_model(model_path)
        detector.encoder = detector._build_encoder_from_loaded_model()

        return detector

    def get_params(self) -> dict[str, Any]:
        """Devuelve la configuración necesaria para reconstruir el detector."""
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "dropout_rate": self.dropout_rate,
            "batch_norm": self.batch_norm,
            "l2_reg": self.l2_reg,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "reconstruction_error": self.reconstruction_error,
        }

    def _build_encoder_from_loaded_model(self) -> tf.keras.Model:
        latent_layer = self.model.get_layer("latent")
        return tf.keras.Model(
            inputs=self.model.input,
            outputs=latent_layer.output,
            name="encoder",
        )

    def _collect_inputs(self, data: Any) -> np.ndarray:
        """Extrae X como np.ndarray desde np.ndarray o tf.data.Dataset."""
        if isinstance(data, np.ndarray):
            return data.astype(np.float32, copy=False)

        if isinstance(data, tf.data.Dataset):
            batches: list[np.ndarray] = []

            for batch in data:
                if isinstance(batch, tuple):
                    x_batch = batch[0]
                else:
                    x_batch = batch

                if isinstance(x_batch, tf.Tensor):
                    x_batch = x_batch.numpy()

                batches.append(np.asarray(x_batch, dtype=np.float32))

            if not batches:
                raise ValueError("Cannot score an empty tf.data.Dataset.")

            return np.concatenate(batches, axis=0)

        raise TypeError(
            "Unsupported data type. Expected np.ndarray or tf.data.Dataset, "
            f"but got {type(data).__name__}."
        )