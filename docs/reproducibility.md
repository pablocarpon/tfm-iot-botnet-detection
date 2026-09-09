# Reproducción de los experimentos / Reproducing the experiments

[Español](#español) · [English](#english)

## Español

Esta guía detalla cómo preparar los datos y repetir los experimentos conservando los resultados originales. Completa primero la [instalación del proyecto](../README.md#instalación-y-ejecución) y ejecuta los comandos desde la raíz del repositorio.

### 1. Preparar N-BaIoT

```bash
python scripts/prepare_n_baiot_data.py
```

Este comando necesita acceso a Kaggle y descarga los archivos mediante `kagglehub`. Genera `base/`, `splits/per_device/`, `splits/global/`, `scaled/per_device/` y `scaled/global/` bajo `data/processed/n_baiot/`, además de los escaladores en `outputs/models/n_baiot/scalers/`. Debe reservarse espacio para la caché de descarga y las distintas versiones procesadas de un dataset de varios gigabytes. Volver a ejecutarlo regenera esos datos y escaladores.

### 2. Crear configuraciones para una nueva ejecución

Las rutas de salida originales ya contienen los resultados del TFM y los YAML tienen `outputs.overwrite: false`. Por ello, ejecutar directamente una etapa con su configuración original produce `FileExistsError` si la salida ya existe.

El siguiente bloque crea copias de las ocho configuraciones con las salidas y referencias entre etapas dirigidas a `outputs/reproduction/runs/`:

```bash
python - <<'PY'
from pathlib import Path
import yaml

config_dir = Path("configs/reproduction")
config_dir.mkdir(parents=True, exist_ok=False)
for source in sorted(Path("configs").glob("nbaiot_*.yaml")):
    config = yaml.safe_load(source.read_text())
    config["outputs"]["output_dir"] = config["outputs"]["output_dir"].replace(
        "outputs/runs/", "outputs/reproduction/runs/", 1
    )
    for key in ("stage_1_run_dir", "stage_2_run_dir"):
        if key in config["paths"]:
            config["paths"][key] = config["paths"][key].replace(
                "outputs/runs/", "outputs/reproduction/runs/", 1
            )
    (config_dir / source.name).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
PY
```

Para una ejecución posterior, utilizar otros directorios de configuración y salida. Las rutas de los YAML se resuelven desde el directorio de trabajo, por lo que los scripts deben lanzarse desde la raíz del repositorio.

### 3. Entrenar y optimizar

```bash
# LOF: optimización y entrenamiento final por dispositivo
python scripts/run_nbaiot_lof_per_device.py --stage 1 --config configs/reproduction/nbaiot_lof_per_device_stage_1.yaml

# Autoencoders por dispositivo: comparación inicial y optimización
python scripts/run_nbaiot_autoencoder_per_device.py --stage 1 --config configs/reproduction/nbaiot_autoencoder_per_device_stage_1.yaml
python scripts/run_nbaiot_autoencoder_per_device.py --stage 2 --config configs/reproduction/nbaiot_autoencoder_per_device_stage_2.yaml

# Autoencoder global: comparación inicial y optimización
python scripts/run_nbaiot_autoencoder_global.py --stage 1 --config configs/reproduction/nbaiot_autoencoder_global_stage_1.yaml
python scripts/run_nbaiot_autoencoder_global.py --stage 2 --config configs/reproduction/nbaiot_autoencoder_global_stage_2.yaml
```

En los autoencoders, la etapa 1 permite justificar la arquitectura base; la etapa 2 utiliza la arquitectura fijada en su propio YAML. La selección entre ambas etapas es una decisión experimental documentada en los notebooks de optimización.

### 4. Calibrar y evaluar

Tras un nuevo entrenamiento, volver a calcular los P99 con el [notebook de umbrales](../notebooks/02_models_experimentation/anomaly_th_study.ipynb), apuntando sus rutas de resultados a `../../outputs/reproduction/runs`. Trasladar los valores a `thresholds` en las configuraciones de evaluación por dispositivo y a `threshold.value` en la global, dentro de `configs/reproduction/`.

**Los umbrales originales corresponden a los modelos guardados del TFM; no se recalculan automáticamente al reentrenar.** Esta calibración debe utilizar exclusivamente las nuevas puntuaciones de validación benigna.

```bash
python scripts/run_nbaiot_lof_per_device.py --stage 2 --config configs/reproduction/nbaiot_lof_per_device_stage_2.yaml
python scripts/run_nbaiot_autoencoder_per_device.py --stage 3 --config configs/reproduction/nbaiot_autoencoder_per_device_stage_3.yaml
python scripts/run_nbaiot_autoencoder_global.py --stage 3 --config configs/reproduction/nbaiot_autoencoder_global_stage_3.yaml
```

Las etapas de evaluación generan `test_predictions.parquet` y resúmenes JSON. Para analizar la nueva ejecución, ajustar también las rutas de los notebooks de resultados a `../../outputs/reproduction/runs`.

El proyecto fija semillas y conserva las configuraciones, pero las dependencias se declaran con versiones mínimas y no existe un archivo de bloqueo del entorno. Las versiones de las bibliotecas, el hardware y la ejecución de TensorFlow pueden introducir diferencias al repetir el entrenamiento. Los resultados incluidos permiten consultar la ejecución de referencia sin reentrenar los modelos.

---

## English

This guide explains how to prepare the data and repeat the experiments while preserving the original results. First complete the [project installation](../README.md#installation-and-execution), then run the commands from the repository root.

### 1. Prepare N-BaIoT

```bash
python scripts/prepare_n_baiot_data.py
```

This command requires access to Kaggle and downloads files through `kagglehub`. It creates `base/`, `splits/per_device/`, `splits/global/`, `scaled/per_device/` and `scaled/global/` under `data/processed/n_baiot/`, plus scalers under `outputs/models/n_baiot/scalers/`. Allow disk space for the download cache and multiple processed versions of a dataset of several gigabytes. Rerunning the command regenerates these data files and scalers.

### 2. Create configurations for a new run

The original output paths already contain the thesis results, and the YAML files set `outputs.overwrite: false`. Running a stage directly with its original configuration therefore raises `FileExistsError` if its output directory exists.

The following block copies all eight configurations and redirects outputs and references between stages to `outputs/reproduction/runs/`:

```bash
python - <<'PY'
from pathlib import Path
import yaml

config_dir = Path("configs/reproduction")
config_dir.mkdir(parents=True, exist_ok=False)
for source in sorted(Path("configs").glob("nbaiot_*.yaml")):
    config = yaml.safe_load(source.read_text())
    config["outputs"]["output_dir"] = config["outputs"]["output_dir"].replace(
        "outputs/runs/", "outputs/reproduction/runs/", 1
    )
    for key in ("stage_1_run_dir", "stage_2_run_dir"):
        if key in config["paths"]:
            config["paths"][key] = config["paths"][key].replace(
                "outputs/runs/", "outputs/reproduction/runs/", 1
            )
    (config_dir / source.name).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
PY
```

Use different configuration and output directories for subsequent runs. YAML paths are resolved from the working directory, so launch the scripts from the repository root.

### 3. Train and optimise

```bash
# LOF: per-device optimisation and final training
python scripts/run_nbaiot_lof_per_device.py --stage 1 --config configs/reproduction/nbaiot_lof_per_device_stage_1.yaml

# Per-device autoencoders: initial comparison and optimisation
python scripts/run_nbaiot_autoencoder_per_device.py --stage 1 --config configs/reproduction/nbaiot_autoencoder_per_device_stage_1.yaml
python scripts/run_nbaiot_autoencoder_per_device.py --stage 2 --config configs/reproduction/nbaiot_autoencoder_per_device_stage_2.yaml

# Global autoencoder: initial comparison and optimisation
python scripts/run_nbaiot_autoencoder_global.py --stage 1 --config configs/reproduction/nbaiot_autoencoder_global_stage_1.yaml
python scripts/run_nbaiot_autoencoder_global.py --stage 2 --config configs/reproduction/nbaiot_autoencoder_global_stage_2.yaml
```

For autoencoders, stage 1 supports the choice of base architecture; stage 2 uses the architecture specified in its own YAML. Selection between these stages is an experimental decision documented in the optimisation notebooks.

### 4. Calibrate and evaluate

After retraining, recalculate the P99 thresholds using the [threshold notebook](../notebooks/02_models_experimentation/anomaly_th_study.ipynb), pointing its result paths to `../../outputs/reproduction/runs`. Copy the values into `thresholds` for the per-device evaluation configurations and `threshold.value` for the global configuration, under `configs/reproduction/`.

**The original thresholds correspond to the saved thesis models; they are not recalculated automatically after retraining.** Calibration must use only the new benign validation scores.

```bash
python scripts/run_nbaiot_lof_per_device.py --stage 2 --config configs/reproduction/nbaiot_lof_per_device_stage_2.yaml
python scripts/run_nbaiot_autoencoder_per_device.py --stage 3 --config configs/reproduction/nbaiot_autoencoder_per_device_stage_3.yaml
python scripts/run_nbaiot_autoencoder_global.py --stage 3 --config configs/reproduction/nbaiot_autoencoder_global_stage_3.yaml
```

Evaluation stages produce `test_predictions.parquet` files and JSON summaries. To analyse the new run, also point the results notebooks to `../../outputs/reproduction/runs`.

The project sets seeds and preserves configurations, but dependencies use minimum version constraints and there is no environment lockfile. Library versions, hardware and TensorFlow execution can introduce differences when training is repeated. Included results allow inspection of the reference run without retraining.
