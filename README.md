# Detección de botnets IoT mediante aprendizaje automático

[Español](#español) · [English](#english)

## Español

Repositorio del TFM **«Estudio comparativo de modelos de aprendizaje automático para la detección de intrusiones en estructuras de IoT urbanas»**.

**Autores:** Carlos García Checa, Isabel Llamas Ortega y Pablo Calvo Arpón.

**Titulación:** Máster en Inteligencia Artificial — Universidad Internacional de La Rioja (UNIR).

El proyecto compara **LOF por dispositivo, autoencoders por dispositivo y un autoencoder global** para estudiar el equilibrio entre detección de ataques y coste de mantener los modelos en una red IoT. El entrenamiento y la calibración utilizan exclusivamente tráfico benigno, con un protocolo de particionado y preprocesado común.

**Tecnologías:** Python · scikit-learn · TensorFlow/Keras · Optuna · pandas · PyArrow/Parquet · Jupyter · YAML.

### Datos

Se utiliza **N-BaIoT**, con **7.062.606 muestras**, **115 características** y tráfico de **nueve dispositivos IoT**, incluyendo ataques de Mirai y Gafgyt. El pipeline descarga la [distribución de Kaggle](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset) y genera los datos locales en `data/processed/n_baiot/`, excluidos de Git.

### Resultados

Evaluación conjunta sobre los nueve dispositivos presentes en entrenamiento, con umbrales P99 calibrados sobre validación benigna. ROC-AUC resume la discriminación entre clases; recall, FPR y MCC corresponden al umbral seleccionado.

| Estrategia | ROC-AUC | Recall | FPR | MCC | Modelos | Tamaño total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LOF por dispositivo | 0.8664 | 0.7531 | 0.0100 | 0.2179 | 9 | 187.91 MiB |
| AE por dispositivo | **0.9924** | 0.7530 | 0.0103 | 0.2178 | 9 | 2.85 MiB |
| AE global | 0.9847 | 0.7224 | 0.0098 | 0.2019 | **1** | **0.49 MiB** |

El autoencoder por dispositivo alcanza la mayor capacidad discriminativa; el global conserva un rendimiento próximo con un único modelo y menor almacenamiento. Los notebooks desarrollan el análisis por dispositivo y ataque, el efecto de la prevalencia y las conclusiones.

### Organización

```text
tfm-iot-botnet-detection/
├── configs/          # Configuración YAML de los experimentos
├── data/             # Datos locales, excluidos de Git en raw/ y processed/
├── docs/             # Documentación y guía de reproducción
├── notebooks/        # Exploración, experimentación y resultados
├── outputs/          # Modelos, escaladores, historiales y predicciones
├── scripts/          # Preparación de datos y ejecución por etapas
├── src/iotbotnet/     # Paquete Python: datos, modelos y experimentos
└── pyproject.toml    # Definición del paquete y dependencias
```

### Notebooks

| Contenido | Cuadernos |
| --- | --- |
| Exploración de datos | [EDA de N-BaIoT](notebooks/01_datasets_analysis/EDA_N_BaIoT.ipynb) |
| Selección y optimización de arquitecturas | [AE por dispositivo](notebooks/02_models_experimentation/autoencoder_per_device_opt_study.ipynb) · [AE global](notebooks/02_models_experimentation/autoencoder_global_opt_study.ipynb) |
| Calibración | [Estudio de umbrales](notebooks/02_models_experimentation/anomaly_th_study.ipynb) |
| Resultados individuales | [LOF](notebooks/03_results_analysis/LOF_results_analysis.ipynb) · [AE por dispositivo](notebooks/03_results_analysis/AE_per_device_results_analysis.ipynb) · [AE global](notebooks/03_results_analysis/AE_global_results_analysis.ipynb) |
| **Comparativa y conclusiones** | **[Análisis comparativo](notebooks/03_results_analysis/comparative_analysis.ipynb)** |

### Instalación y ejecución

Requiere **Python 3.11 o superior**. Desde la raíz del repositorio, en Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks]" "kagglehub[pandas-datasets]" tqdm
jupyter notebook notebooks/03_results_analysis/comparative_analysis.ipynb
```

Los notebooks incluyen salidas guardadas y el análisis puede repetirse con las predicciones de `outputs/runs/`, sin reentrenar. Sus rutas son relativas a la carpeta del cuaderno; ejecutar el análisis completo requiere cargar millones de predicciones en memoria.

Para descargar y preparar los datos:

```bash
python scripts/prepare_n_baiot_data.py
```

La [guía de reproducción](docs/reproducibility.md#español) explica cómo configurar una nueva ejecución, entrenar, recalibrar los umbrales y evaluar. Las salidas originales están protegidas frente a sobrescritura mediante la configuración de los experimentos.

---

## English

### IoT botnet detection using machine learning

Repository for the Master's Thesis **“Comparative study of machine learning models for intrusion detection in urban IoT infrastructures”**. The official Spanish title is provided above.

**Authors:** Carlos García Checa, Isabel Llamas Ortega and Pablo Calvo Arpón.

**Degree:** Master's Degree in Artificial Intelligence — Universidad Internacional de La Rioja (UNIR).

The project compares **per-device LOF, per-device autoencoders and a global autoencoder** to study the balance between attack detection and model maintenance costs in an IoT network. Training and calibration use only benign traffic, with a common data splitting and preprocessing protocol.

**Technologies:** Python · scikit-learn · TensorFlow/Keras · Optuna · pandas · PyArrow/Parquet · Jupyter · YAML.

### Dataset

The project uses **N-BaIoT**, with **7,062,606 samples**, **115 features** and traffic from **nine IoT devices**, including Mirai and Gafgyt attacks. The pipeline downloads the [Kaggle distribution](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset) and generates local data under `data/processed/n_baiot/`, which is excluded from Git.

### Results

Pooled evaluation across the nine devices represented in training, with P99 thresholds calibrated on benign validation data. ROC-AUC summarises class discrimination; recall, FPR and MCC refer to the selected operating threshold.

| Strategy | ROC-AUC | Recall | FPR | MCC | Models | Total size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Per-device LOF | 0.8664 | 0.7531 | 0.0100 | 0.2179 | 9 | 187.91 MiB |
| Per-device AE | **0.9924** | 0.7530 | 0.0103 | 0.2178 | 9 | 2.85 MiB |
| Global AE | 0.9847 | 0.7224 | 0.0098 | 0.2019 | **1** | **0.49 MiB** |

The per-device autoencoder achieves the strongest discrimination; the global model retains similar performance with a single model and a smaller storage footprint. The notebooks cover device- and attack-level analysis, the effect of prevalence and the conclusions.

### Structure

```text
tfm-iot-botnet-detection/
├── configs/          # YAML experiment configurations
├── data/             # Local data; raw/ and processed/ excluded from Git
├── docs/             # Documentation and reproduction guide
├── notebooks/        # Exploration, experiments and results
├── outputs/          # Models, scalers, histories and predictions
├── scripts/          # Data preparation and staged execution
├── src/iotbotnet/     # Python package: data, models and experiments
└── pyproject.toml    # Package definition and dependencies
```

### Analysis notebooks

The notebooks and their detailed discussion are written in Spanish.

| Contents | Notebooks |
| --- | --- |
| Data exploration | [N-BaIoT EDA](notebooks/01_datasets_analysis/EDA_N_BaIoT.ipynb) |
| Architecture selection and optimisation | [Per-device AE](notebooks/02_models_experimentation/autoencoder_per_device_opt_study.ipynb) · [Global AE](notebooks/02_models_experimentation/autoencoder_global_opt_study.ipynb) |
| Calibration | [Threshold study](notebooks/02_models_experimentation/anomaly_th_study.ipynb) |
| Individual results | [LOF](notebooks/03_results_analysis/LOF_results_analysis.ipynb) · [Per-device AE](notebooks/03_results_analysis/AE_per_device_results_analysis.ipynb) · [Global AE](notebooks/03_results_analysis/AE_global_results_analysis.ipynb) |
| **Comparison and conclusions** | **[Comparative analysis](notebooks/03_results_analysis/comparative_analysis.ipynb)** |

### Installation and execution

Requires **Python 3.11 or later**. Run from the repository root in Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks]" "kagglehub[pandas-datasets]" tqdm
jupyter notebook notebooks/03_results_analysis/comparative_analysis.ipynb
```

Notebooks include saved outputs, and the analysis can be repeated using predictions in `outputs/runs/` without retraining. Paths are relative to each notebook's directory; running the full analysis loads millions of predictions into memory.

To download and prepare the data:

```bash
python scripts/prepare_n_baiot_data.py
```

The [reproduction guide](docs/reproducibility.md#english) explains how to configure a new run, train, recalibrate thresholds and evaluate. Experiment configurations protect the original outputs from being overwritten.
