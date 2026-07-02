from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class LOFAnomalyDetector:
    """Detector de anomalías basado en Local Outlier Factor.

    Diseñado para entrenarse únicamente con tráfico benigno y puntuar
    posteriormente muestras nuevas de validación o test.

    Por diseño, usa siempre novelty=True.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "minkowski",
        n_jobs: int | None = -1,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs

        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            metric=metric,
            novelty=True,
            n_jobs=n_jobs,
        )

    def fit(self, data: np.ndarray, **kwargs: Any) -> "LOFAnomalyDetector":
        """Entrena LOF usando muestras benignas."""
        self.model.fit(data)
        return self

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Devuelve un score de anomalía por muestra.

        Convención del proyecto:
        - score alto = muestra más anómala

        scikit-learn devuelve valores más bajos para muestras más anómalas,
        por eso se invierte el signo.
        """
        scores = self.model.score_samples(data)
        return -scores

    def save(self, path: str | Path) -> None:
        """Guarda el detector completo en disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "LOFAnomalyDetector":
        """Carga un detector guardado previamente."""
        path = Path(path)
        loaded_model = joblib.load(path)

        if not isinstance(loaded_model, cls):
            raise TypeError(
                f"Expected object of type {cls.__name__}, "
                f"but got {type(loaded_model).__name__}."
            )

        return loaded_model

    def get_params(self) -> dict[str, Any]:
        """Devuelve los hiperparámetros principales del detector."""
        return {
            "type": "lof",
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "n_jobs": self.n_jobs,
            "novelty": True,
        }