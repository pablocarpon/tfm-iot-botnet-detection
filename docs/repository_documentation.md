# Repository Documentation

## 1. Objetivo del repositorio

Este repositorio contiene el código, la configuración experimental, la documentación y los resultados necesarios para desarrollar los experimentos del TFM:

> Estudio comparativo de modelos de aprendizaje automático para la detección de intrusiones en estructuras IoT urbanas.

La estructura se ha diseñado para soportar la comparación de distintas estrategias de detección:

- LOF por dispositivo.
- Autoencoder por dispositivo.
- Autoencoder global.
- Autoencoder global con adaptación ligera.

Asimismo, la estructura permite trabajar con múltiples datasets manteniendo separadas las distintas responsabilidades del proyecto:

- Datos.
- Configuración experimental.
- Implementación.
- Resultados.
- Documentación.

---

## 2. Criterios utilizados para diseñar la estructura

La estructura del repositorio se diseñó para cubrir cuatro necesidades concretas del TFM:

1. Separar datos, código, configuración y resultados.
2. Permitir comparar múltiples estrategias experimentales bajo una mismas condiciones.
3. Facilitar la incorporación de nuevos datasets sin modificar la estructura existente.
4. Mantener el repositorio ligero evitando almacenar datasets de gran tamaño.

---

## 3. Estructura general del repositorio

```text
tfm-iot-botnet-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
├── data/
├── notebooks/
├── src/
├── scripts/
├── outputs/
└── docs/
```

---

## 4. Descripción detallada de los directorios

---

### configs/

**Estado:** Actual

#### Propósito

Contiene los archivos de configuración utilizados para definir datasets, modelos y experimentos.

La idea principal es que la definición de un experimento esté separada de la implementación del mismo.

Esto permite modificar configuraciones sin necesidad de cambiar código.

#### Contenido esperado

```text
configs/
└── experiments/
```

#### Ejemplo

```text
configs/experiments/
└── nbaiot_global_adapted.yaml
```

```yaml
dataset: nbaiot
strategy: global_adapted_autoencoder

model:
  latent_dim: 16
  epochs: 100

split:
  leave_one_device_out: Danmini_Doorbell
```

---

### data/

**Estado:** Actual

#### Propósito

Contiene los datos utilizados durante la experimentación.

Este directorio se divide en tres partes claramente diferenciadas:

- Datos originales.
- Datos procesados.
- Metadatos del proyecto.

```text
data/
├── raw/
├── processed/
└── metadata/
```

---

### data/raw/

**Estado:** Actual

#### Propósito

Almacenar localmente los datasets originales descargados.

Los datasets utilizados en este TFM son demasiado grandes para almacenarlos dentro del repositorio Git.

Por este motivo:

- No se subirán a GitHub.
- No se versionarán con Git.
- Se ignorarán mediante `.gitignore`.

Cada miembro del equipo podrá utilizar este directorio para disponer de una copia local de los datasets.

#### Contenido esperado

```text
data/raw/
├── nbaiot/
└── ciciot2023/
```

#### Ejemplo

```text
data/raw/nbaiot/
├── Danmini_Doorbell/
├── Ecobee_Thermostat/
├── Provision_PT_737E/
└── ...
```

---

### data/processed/

**Estado:** Actual

#### Propósito

Almacenar versiones procesadas de los datasets.

Su función es evitar repetir tareas costosas de preprocesado cada vez que se ejecuta un experimento.

Al igual que `raw/`, este directorio:

- No se subirá a GitHub.
- No se versionará con Git.
- Se ignorará mediante `.gitignore`.

#### Contenido esperado

```text
data/processed/
├── nbaiot/
└── ciciot2023/
```

#### Ejemplo

```text
data/processed/nbaiot/
├── scaled.parquet
├── train_split.parquet
└── validation_split.parquet
```

---

### data/metadata/

**Estado:** Actual

#### Propósito

Contiene información ligera necesaria para reproducir los experimentos.

A diferencia de `raw/` y `processed/`, estos archivos sí forman parte del repositorio.

#### Contenido esperado

```text
data/metadata/
├── attack_mapping.yaml
├── nbaiot_devices.csv
└── ciciot_functional_groups.csv
```

#### Ejemplo

```csv
device_id,device_name,device_type
1,Danmini_Doorbell,doorbell
2,Ecobee_Thermostat,thermostat
```

---

### notebooks/

**Estado:** Actual

#### Propósito

Contiene los cuadernos utilizados para:

- Análisis exploratorio.
- Visualización.
- Pruebas rápidas.
- Análisis de resultados.

Los notebooks no deben contener la implementación principal del proyecto.

La lógica reutilizable debe residir en `src`.

#### Contenido esperado

```text
notebooks/
├── 01_dataset_analysis/
├── 02_models_experimentation/
└── 03_results_analysis/
```

#### Ejemplo

```text
01_dataset_analysis/
└── EDA_N_BaIoT.ipynb
```

---

### src/

**Estado:** Actual

#### Propósito

Contiene la implementación principal del software experimental.

Todo el código reutilizable del proyecto debe vivir aquí.

```text
src/
├── data/
├── models/
├── adaptation/
├── evaluation/
├── experiments/
└── utils/
```

---

### src/data/

**Estado:** Actual

#### Propósito

Contiene el código encargado de:

- Cargar datasets.
- Preprocesar datos.
- Generar particiones experimentales.

#### Contenido esperado

```text
src/data/
├── loaders.py
├── preprocessing.py
└── splits.py
```

#### Ejemplo

```python
load_nbaiot()
load_ciciot2023()

scale_features()

leave_one_device_out_split()
```

#### Cuándo modificar este directorio

- Cuando se añada un nuevo dataset.
- Cuando se cambie el pipeline de preprocesado.
- Cuando aparezca una nueva estrategia de particionado.

---

### src/models/

**Estado:** Actual

#### Propósito

Contiene las implementaciones de los modelos evaluados durante el TFM.

#### Contenido esperado

```text
src/models/
├── autoencoder.py
├── lof.py
└── factory.py
```

#### Ejemplo

```python
model = Autoencoder(...)
```

o

```python
model = LocalOutlierFactor(...)
```

#### Cuándo modificar este directorio

- Cuando se implemente una nueva estrategia de detección.
- Cuando se modifique una arquitectura existente.

---

### src/adaptation/

**Estado:** Previsto

#### Propósito

Contendrá los mecanismos de adaptación ligera propuestos en el TFM.

La propuesta experimental plantea que un modelo global pueda complementarse mediante estadísticas específicas de dispositivo o grupo funcional.

Este directorio contendrá dicha lógica.

#### Contenido esperado

```text
src/adaptation/
├── centroids.py
├── thresholds.py
└── scoring.py
```

#### Ejemplo

```python
compute_device_centroids()
compute_thresholds()
compute_adaptation_score()
```

#### Cuándo crear este directorio

Cuando comiencen los experimentos relacionados con:

- Adaptación ligera.
- Centroides latentes.
- Umbrales específicos por dispositivo.
- Umbrales específicos por grupo funcional.

---

### src/evaluation/

**Estado:** Actual

#### Propósito

Contiene el código utilizado para evaluar los experimentos.

#### Contenido esperado

```text
src/evaluation/
├── metrics.py
├── scalability.py
└── generalization.py
```

#### Ejemplo

Métricas clásicas:

```python
accuracy
precision
recall
f1_score
roc_auc
```

Métricas de escalabilidad:

```python
training_time
num_models
memory_usage
```

#### Cuándo modificar este directorio

Cuando se incorporen nuevas métricas de evaluación.

---

### src/experiments/

**Estado:** Actual

#### Propósito

Contiene la lógica común de ejecución de experimentos.

Su objetivo es evitar duplicar el mismo flujo de trabajo en múltiples notebooks.

#### Contenido esperado

```text
src/experiments/
├── runner.py
└── registry.py
```

#### Responsabilidades

Un experimento típico consiste en:

1. Leer configuración.
2. Cargar datos.
3. Generar particiones.
4. Entrenar modelo.
5. Evaluar resultados.
6. Guardar artefactos.

Esta secuencia debe implementarse aquí.

#### Ejemplo

```python
run_experiment(config)
```

---

### src/utils/

**Estado:** Actual

#### Propósito

Contiene utilidades reutilizables que pueden ser utilizadas desde cualquier módulo.

#### Contenido esperado

```text
src/utils/
├── io.py
├── logging.py
└── random.py
```

#### Ejemplo

```python
save_json()
load_yaml()
set_seed()
```

---

### scripts/

**Estado:** Previsto

#### Propósito

Contendrá puntos de entrada sencillos para ejecutar procesos completos desde línea de comandos.

Toda la lógica seguirá residiendo en `src`.

#### Contenido esperado

```text
scripts/
└── run_experiment.py
```

#### Ejemplo

```bash
python scripts/run_experiment.py \
    --config configs/experiments/nbaiot_global.yaml
```

#### Cuándo crear este directorio

Cuando ejecutar experimentos desde notebooks deje de ser cómodo o empiecen a repetirse tareas de ejecución.

---

### outputs/

**Estado:** Actual

#### Propósito

Contiene todos los artefactos generados durante la experimentación.

```text
outputs/
├── figures/
├── tables/
├── models/
└── runs/
```

---

### outputs/figures/

#### Propósito

Almacenar figuras utilizadas durante el análisis y la redacción de la memoria.

#### Ejemplo

```text
roc_curve.png
f1_comparison.png
```

---

### outputs/tables/

#### Propósito

Almacenar tablas de resultados.

#### Ejemplo

```text
strategy_comparison.csv
generalization_results.csv
```

---

### outputs/models/

#### Propósito

Almacenar modelos entrenados que se desee conservar.

#### Ejemplo

```text
global_autoencoder.pt
```

---

### outputs/runs/

#### Propósito

Almacenar resultados asociados a una ejecución concreta.

Permite reconstruir exactamente cómo se obtuvo un resultado.

#### Ejemplo

```text
outputs/runs/
└── 2026_06_15_global_adapted/
    ├── config.yaml
    ├── metrics.json
    ├── predictions.csv
    └── model.pt
```

---

### docs/

**Estado:** Actual

#### Propósito

Contiene documentación del proyecto.

#### Contenido esperado

```text
docs/
├── repository_documentation.md
├── experiment_design.md
└── reproducibility.md
```

#### Cuándo modificar este directorio

Cuando cambien decisiones importantes relacionadas con:

- estructura del repositorio,
- diseño experimental,
- reproducción de resultados.

---

## 5. Flujo de un experimento

Ejemplo:

**Autoencoder global con adaptación ligera sobre N-BaIoT.**

### Paso 1

Se carga el dataset en memoria mediante alguna API o el dataset se almacena localmente en:

```text
data/raw/nbaiot/
```

### Paso 2

Se ejecuta el pipeline de preprocesado.

Los resultados se almacenan en:

```text
data/processed/nbaiot/
```

### Paso 3

Se define la configuración experimental.

```text
configs/experiments/
└── nbaiot_global_adapted.yaml
```

### Paso 4

`src/data/` carga los datos y genera las particiones necesarias.

### Paso 5

`src/models/` entrena el modelo correspondiente.

### Paso 6

`src/adaptation/` aplica la adaptación ligera.

### Paso 7

`src/evaluation/` calcula las métricas.

### Paso 8

Los resultados se almacenan en:

```text
outputs/
```

### Paso 9

Los notebooks se utilizan para analizar los resultados obtenidos.

---

## 6. Incorporación de nuevos datasets

Para añadir un nuevo dataset deben realizarse únicamente tres acciones:

1. Almacenar el dataset localmente en `data/raw/` o en algún repositorio directamente accesible mediante código.
2. Implementar el cargador correspondiente en `src/data/loaders.py`.
3. Añadir la configuración correspondiente en `configs/datasets/`.

No debería ser necesario modificar el resto de la estructura del repositorio.