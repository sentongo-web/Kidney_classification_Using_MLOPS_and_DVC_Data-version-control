---
title: KidneyDL CT Scan Classifier
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
---

## KidneyAI: End-to-End Kidney CT Scan Classification with MLOps

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline%20Versioning-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Remote%20Tracking-FF6B35?logoColor=white)](https://dagshub.com/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## What This Project Is

This is a production-style, end-to-end machine learning project that classifies kidney CT scan images as either **Normal** or **Tumor**. But the model itself is only one piece of the story. The real focus of this project is everything that surrounds it: a fully reproducible DVC pipeline, experiment tracking with MLflow and DagsHub, a clean configuration-driven codebase, a Flask web application, and a Dockerised deployment setup.

It was built to demonstrate what a real MLOps workflow looks like in practice, not just the notebook that produces a metric, but the entire system that allows a model to be trained, evaluated, versioned, and served reliably.

---

## The Problem

Kidney disease is among the leading causes of death globally, and it often goes undetected until its later stages when treatment options become limited. Radiologists manually reviewing CT scans are under enormous pressure, and any tool that can reliably flag suspicious scans for closer attention has genuine clinical value.

This project builds a binary image classifier that can look at a kidney CT scan and tell you, within seconds, whether the kidney appears normal or shows signs of a tumor. It is trained on a labelled CT scan dataset and achieves approximately **89.9% validation accuracy** using a fine-tuned VGG16 network.

---

## Why VGG16?

VGG16 was selected deliberately, not arbitrarily. Here is the reasoning:

Its architecture is built from uniform 3x3 convolutional layers stacked into increasing depth. This design is especially good at learning fine-grained local textures, which is critical in medical imaging where the difference between healthy and abnormal tissue often comes down to subtle structural patterns rather than large-scale shape differences.

Pre-trained on ImageNet, VGG16 already knows how to see. Its lower layers encode general-purpose feature detectors for edges, corners, and textures. Those weights do not need to be learned from scratch. Only the top classification layers need to be adapted to the kidney scan domain, which means the model can achieve strong performance with far less labelled data than training from scratch would require.

It is also a stable, well-understood architecture. In a medical context, that matters. The behaviour of the model is predictable, and the features it learns can be interpreted through tools like Grad-CAM.

---

## Model Performance

| Metric   | Value  |
|----------|--------|
| Accuracy | 89.9%  |
| Loss     | 1.26   |

Metrics are logged automatically to MLflow after every pipeline run. You can view all experiment runs, compare parameters, and download model artifacts directly from the DagsHub MLflow UI.

---

## Project Structure

```text
Kidney_classification_Using_MLOPS_and_DVC/
│
├── config/
│   └── config.yaml                  Central path and artifact configuration
│
├── params.yaml                      All model hyperparameters in one place
├── dvc.yaml                         DVC pipeline stage definitions
├── dvc.lock                         DVC lock file tracking stage state
├── main.py                          Runs all pipeline stages sequentially
├── app.py                           Flask web application
├── Dockerfile                       Container definition for the prediction server
├── requirements.txt                 Python dependencies
├── setup.py                         Installable package definition
├── scores.json                      Latest evaluation metrics
│
├── src/cnnClassifier/
│   ├── __init__.py                  Logger setup
│   ├── constants/                   Project-wide constants (config file paths)
│   ├── entity/
│   │   └── config_entity.py         Typed dataclasses for each pipeline stage config
│   ├── config/
│   │   └── configuration.py         ConfigurationManager: reads YAML and builds configs
│   ├── utils/
│   │   └── common.py                Shared utilities: YAML reading, directory creation, JSON saving
│   ├── components/
│   │   ├── data_ingestion.py         Downloads and extracts the dataset
│   │   ├── prepare_base_model.py     Loads VGG16 and adds the classification head
│   │   ├── model_trainer.py          Trains the model with augmentation support
│   │   └── model_evaluation_mlflow.py Evaluates and logs to MLflow via DagsHub
│   └── pipeline/
│       ├── stage_01_data_ingestion.py
│       ├── stage_02_prepare_base_model.py
│       ├── stage_03_model_trainer.py
│       ├── stage_04_model_evaluation.py
│       └── prediction.py             Prediction pipeline used by the Flask app
│
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_trainer.ipynb
│   └── 04_model_evaluation.ipynb     Each stage was prototyped here first
│
└── templates/
    └── index.html                    Web UI for the prediction app
```

---

## The ML Pipeline

The pipeline has four stages, each defined in `dvc.yaml` and executed in order by DVC.

```text
Stage 1          Stage 2                  Stage 3           Stage 4
Data Ingestion   Base Model Preparation   Model Training    Model Evaluation
```

### Stage 1: Data Ingestion

Downloads the kidney CT scan dataset from Google Drive using `gdown`, extracts the zip archive, and places the images into the `artifacts/data_ingestion/` directory. DVC tracks the output so this stage is skipped if the data already exists and nothing has changed.

### Stage 2: Base Model Preparation

Loads VGG16 with ImageNet weights and without its top classification layers. Adds a custom head: a global average pooling layer followed by a dense output layer with softmax activation for the two classes, Normal and Tumor. The base VGG16 layers are frozen. The resulting model is saved to disk so the training stage can pick it up.

### Stage 3: Model Training

Loads the prepared base model, recompiles it with an SGD optimiser, and trains it on the kidney CT images. Supports data augmentation (horizontal flip, zoom, shear) to improve generalisation. The trained model is saved as `artifacts/training/model.h5`.

### Stage 4: Model Evaluation

Loads the trained model and evaluates it against the 30 percent validation split. Loss and accuracy are saved to `scores.json` and logged to MLflow. The model is also registered in the MLflow Model Registry under the name `VGG16Model`.

---

## Experiment Tracking with MLflow and DagsHub

All runs are tracked remotely on DagsHub, which acts as the MLflow tracking server. Every time the evaluation stage runs, it logs:

- All hyperparameters from `params.yaml`
- Validation loss and accuracy
- The trained model as an MLflow artifact
- A registered model version in the MLflow Model Registry

You can view the experiment runs at:
[https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow](https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow)

---

## Configuration

Everything is driven by two YAML files. There are no hardcoded paths or hyperparameters anywhere in the source code.

**`config/config.yaml`** manages all file paths and artifact locations:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: "https://drive.google.com/file/d/16PZpADG4Pl_SBr2E3DEcvXsLQ5DSUtDP/view?usp=sharing"
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5

evaluation:
  path_of_model: artifacts/training/model.h5
  training_data: artifacts/data_ingestion/kidney-ct-scan-image
  mlflow_uri: "https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow"
  all_params:
    AUGMENTATION: True
    IMAGE_SIZE: [224, 224, 3]
    BATCH_SIZE: 16
    INCLUDE_TOP: False
    EPOCHS: 5
    CLASSES: 2
    WEIGHTS: imagenet
    LEARNING_RATE: 0.01
```

**`params.yaml`** is where all model hyperparameters live:

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 5
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.git
cd Kidney_classification_Using_MLOPS_and_DVC_Data-version-control
```

### 2. Create and activate a Conda environment

```bash
conda create -n kidney python=3.13 -y
conda activate kidney
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Set up your MLflow credentials

Create a `.env` file in the project root with your DagsHub token:

```env
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

This file is gitignored and will never be committed.

### 5. Run the full pipeline

```bash
dvc repro
```

DVC will execute all four stages in order. If any stage has already run and its inputs have not changed, it will be skipped automatically. After the pipeline finishes, `scores.json` will contain the latest evaluation metrics.

### 6. Launch the web application

```bash
python app.py
```

Open your browser and go to `http://localhost:8080`. You can upload a kidney CT scan image and get a classification result instantly.

### 7. View experiment runs

```bash
mlflow ui
```

Open `http://localhost:5000` to browse all local experiment runs, or visit the DagsHub MLflow URL above to see all remotely tracked runs.

---

## Run with Docker

```bash
docker build -t kidney-classifier .
docker run -p 8080:8080 kidney-classifier
```

Open `http://localhost:8080` in your browser.

---

## The Web Application

The Flask app exposes three routes:

| Route     | Method | Description                                                         |
| --------- | ------ | ------------------------------------------------------------------- |
| `/`       | GET    | Serves the prediction web UI                                        |
| `/predict`| POST   | Accepts an image file and returns the classification result as JSON |
| `/train`  | GET    | Reruns `main.py` to retrain the model from scratch                  |

The prediction endpoint returns a response like this:

```json
[{"image": "Normal"}]
```

or

```json
[{"image": "Tumor"}]
```

The UI supports drag and drop, shows a live preview of the uploaded scan, displays the result with a confidence bar, and works in both light and dark mode with automatic detection of your system preference.

---

## Tech Stack

| Area                | Tools                                             |
| ------------------- | ------------------------------------------------- |
| Deep Learning       | TensorFlow and Keras with VGG16 transfer learning |
| Data Versioning     | DVC                                               |
| Experiment Tracking | MLflow hosted on DagsHub                          |
| Web Framework       | Flask with Flask-CORS                             |
| Data Processing     | NumPy, Pandas, scikit-learn                       |
| Configuration       | PyYAML and python-box                             |
| Package Management  | setuptools with src layout, editable install      |
| Containerisation    | Docker                                            |
| Environment         | Conda with pip                                    |

---

## MLOps Concepts Demonstrated

| Concept                  | How it is implemented                                                           |
| ------------------------ | ------------------------------------------------------------------------------- |
| Data versioning          | DVC tracks the dataset and all model artifacts                                  |
| Pipeline as code         | `dvc.yaml` defines every stage and its dependencies                             |
| Incremental execution    | DVC only reruns stages whose inputs have changed                                |
| Experiment tracking      | MLflow logs parameters, metrics, and model artifacts on every run               |
| Model registry           | Trained models are registered and versioned in the MLflow Model Registry        |
| Configuration management | All paths and hyperparameters live in YAML files with no hardcoded values       |
| Modular ML package       | Source code is structured as an installable Python package                      |
| Reproducibility          | Any contributor can clone the repo and run `dvc repro` to get identical results |
| Containerisation         | Dockerfile ensures the app runs consistently in any environment                 |
| REST API serving         | Flask wraps the prediction pipeline and exposes it over HTTP                    |

---

## About the Author

**Paul Sentongo** is a data scientist and applied AI researcher with a Master's degree in Data Science. He is passionate about building machine learning systems that go beyond the notebook: reproducible, traceable, and deployable. His research interests include deep learning for medical imaging, MLOps infrastructure, and the practical challenges of making AI work in the real world.

Paul is currently open to research positions and industry roles where he can contribute to meaningful AI projects and grow alongside motivated teams.

- GitHub: [github.com/sentongo-web](https://github.com/sentongo-web)
- LinkedIn: [linkedin.com/in/paul-sentongo-885041284](https://www.linkedin.com/in/paul-sentongo-885041284/)
- Email: sentongogray1992@gmail.com
