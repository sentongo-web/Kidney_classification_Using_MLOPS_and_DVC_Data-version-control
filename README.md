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

## KidneyDL: End-to-End Kidney CT Scan Classification with MLOps

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline%20Versioning-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Remote%20Tracking-FF6B35?logoColor=white)](https://dagshub.com/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Sentoz/kidney-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Live Demo

The application is deployed and publicly accessible at:

**[https://huggingface.co/spaces/Sentoz/kidney-classifier](https://huggingface.co/spaces/Sentoz/kidney-classifier)**

Upload a kidney CT scan image and receive an instant classification — Normal or Tumor — directly in your browser. No setup required.

---

## What This Project Is

This is a production-style, end-to-end machine learning project that classifies kidney CT scan images as either **Normal** or **Tumor**. The model is only one piece of the story. The real focus is everything surrounding it: a fully reproducible DVC pipeline, experiment tracking with MLflow and DagsHub, a configuration-driven codebase, a Flask web application, Docker containerisation, and live deployment on Hugging Face Spaces with GitHub Actions CI/CD.

It was built to demonstrate what a genuine MLOps workflow looks like in practice — not just the notebook that produces a metric, but the entire system that allows a model to be trained, evaluated, versioned, and served reliably.

---

## The Problem

Kidney disease is among the leading causes of death globally, and it often goes undetected until its later stages when treatment options become severely limited. Radiologists reviewing CT scans are under enormous pressure, and any tool that reliably flags suspicious scans for closer attention has genuine clinical value.

This project builds a binary image classifier that examines a kidney CT scan and determines, within seconds, whether the kidney appears normal or shows signs of a tumour. Trained on a labelled CT scan dataset and fine-tuned from VGG16, the model achieves approximately **89.9% validation accuracy**.

---

## Why VGG16

VGG16 was selected deliberately. Its architecture is built from uniform 3x3 convolutional layers stacked into increasing depth — a design that is especially strong at learning fine-grained local textures. This matters in medical imaging because the difference between healthy and abnormal tissue often comes down to subtle structural patterns, not large-scale shape differences.

Pre-trained on ImageNet, VGG16 already knows how to see. Its lower layers encode general-purpose feature detectors for edges, corners, and textures. Those weights do not need to be learned from scratch. Only the top classification layers need to be adapted to the kidney scan domain, which means the model achieves strong performance with far less labelled data than training from scratch would require.

It is also a stable, well-understood architecture. In a medical context that matters: the behaviour of the model is predictable, its performance is reproducible, and its features can be interpreted through tools like Grad-CAM if needed.

---

## Model Performance

| Metric | Value |
| --- | --- |
| Accuracy | 89.9% |
| Loss | 1.26 |
| Architecture | VGG16 (fine-tuned) |
| Model Version | v1 — registered in MLflow Model Registry as `VGG16Model` |
| Training Epochs | 5 |
| Optimiser | SGD (learning rate 0.01) |
| Input Size | 224 x 224 x 3 |
| Classes | Normal, Tumor |

Metrics are logged automatically to MLflow after every pipeline run. All experiment runs, parameters, and model artifacts are tracked remotely on DagsHub:

[https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow](https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow)

---

## Project Structure

```text
Kidney_classification_Using_MLOPS_and_DVC/
│
├── .github/
│   └── workflows/
│       └── cd.yml                   GitHub Actions CD pipeline (deploys to HF Spaces on push)
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
├── deploy_to_hf.py                  One-command deployment script for Hugging Face Spaces
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
│   ├── 01_data_ingestion.ipynb       Stage prototyped and validated here first
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_trainer.ipynb
│   └── 04_model_evaluation.ipynb
│
└── templates/
    └── index.html                    Web UI: dark/light mode, drag-and-drop, confidence bar
```

---

## The ML Pipeline

The pipeline has four stages, each defined in `dvc.yaml` and executed in order by DVC. Every stage declares its dependencies and outputs explicitly, so DVC knows exactly which stages to skip and which to rerun when something changes.

```text
Stage 1 → Stage 2 → Stage 3 → Stage 4
Data       Base        Model     Model
Ingestion  Model       Training  Evaluation
           Prep
```

### Stage 1: Data Ingestion

Downloads the kidney CT scan dataset from Google Drive using `gdown`, extracts the zip archive, and places the images into `artifacts/data_ingestion/`. DVC tracks the output so this stage is skipped entirely if the data already exists and nothing has changed.

### Stage 2: Base Model Preparation

Loads VGG16 with ImageNet weights, removes its original classification head, and adds a custom head: global average pooling followed by a dense softmax output layer for two classes (Normal, Tumor). All base VGG16 layers are frozen. Both the base model and the updated model are saved to disk.

### Stage 3: Model Training

Loads the prepared base model and recompiles it with an SGD optimiser. Trains on the kidney CT images with optional data augmentation (horizontal flip, zoom, shear). The trained model is saved to `artifacts/training/model.h5`.

### Stage 4: Model Evaluation

Loads the trained model and evaluates it on the 30 percent validation split. Loss and accuracy are written to `scores.json` and logged to MLflow. The model is also registered in the MLflow Model Registry under the name `VGG16Model` for versioning and downstream use.

---

## CI/CD Pipeline

Every push to `main` on GitHub triggers the GitHub Actions CD workflow defined in `.github/workflows/cd.yml`. The workflow:

1. Checks out the repository with full history and LFS support
2. Pushes the codebase to Hugging Face Spaces via git

Hugging Face then detects the `Dockerfile` and automatically builds and deploys the updated container. The deployment is fully automated after initial setup.

For the initial deployment or when the trained model needs to be included (since it is gitignored), the `deploy_to_hf.py` script uses the Hugging Face Hub Python API to upload all necessary files including the model artifact:

```bash
python deploy_to_hf.py
```

---

## Experiment Tracking with MLflow and DagsHub

All runs are tracked remotely on DagsHub, which acts as the MLflow tracking server. Every time the evaluation stage runs, it logs:

- All hyperparameters from `params.yaml`
- Validation loss and accuracy
- The trained model as an MLflow artifact
- A registered model version in the MLflow Model Registry under `VGG16Model`

View all experiment runs at:
[https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow](https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow)

---

## Configuration

Everything is driven by two YAML files. There are no hardcoded paths or hyperparameters anywhere in the source code.

`config/config.yaml` manages all file paths and artifact locations:

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
```

`params.yaml` is where all model hyperparameters live:

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
conda create -n kidney python=3.10 -y
conda activate kidney
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Set up MLflow credentials

Create a `.env` file in the project root:

```env
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

This file is gitignored and will never be committed.

### 5. Run the full pipeline

```bash
dvc repro
```

DVC executes all four stages in order. Stages whose inputs have not changed are skipped automatically. After the pipeline completes, `scores.json` contains the latest evaluation metrics.

### 6. Launch the web application

```bash
python app.py
```

Open `http://localhost:7860` in your browser. Upload a kidney CT scan image and get a classification result instantly.

### 7. View experiment runs locally

```bash
mlflow ui
```

Open `http://localhost:5000` to browse all local experiment runs, or visit the DagsHub MLflow URL above for all remotely tracked runs.

---

## Run with Docker

```bash
docker build -t kidney-classifier .
docker run -p 7860:7860 kidney-classifier
```

Open `http://localhost:7860` in your browser.

---

## The Web Application

The Flask app exposes three routes:

| Route      | Method | Description                                                          |
|------------|--------|----------------------------------------------------------------------|
| `/`        | GET    | Serves the prediction web UI                                         |
| `/predict` | POST   | Accepts an image file and returns the classification result as JSON  |
| `/train`   | GET    | Reruns `main.py` to retrain the model from scratch                   |

The prediction endpoint returns:

```json
[{"image": "Normal"}]
```

or

```json
[{"image": "Tumor"}]
```

The UI supports drag and drop, shows a live preview of the uploaded scan, displays the result with a confidence bar, and works in both light and dark mode with automatic system preference detection.

---

## Tech Stack

| Area                | Tools                                                |
|---------------------|------------------------------------------------------|
| Deep Learning       | TensorFlow and Keras with VGG16 transfer learning    |
| Data Versioning     | DVC                                                  |
| Experiment Tracking | MLflow hosted on DagsHub                             |
| Web Framework       | Flask with Flask-CORS                                |
| Data Processing     | NumPy, Pandas, scikit-learn                          |
| Configuration       | PyYAML and python-box                                |
| Package Management  | setuptools with src layout, editable install         |
| Containerisation    | Docker                                               |
| Deployment          | Hugging Face Spaces (Docker SDK)                     |
| CI/CD               | GitHub Actions                                       |
| Environment         | Conda with pip                                       |

---

## MLOps Concepts Demonstrated

| Concept | Implementation |
| --- | --- |
| Data versioning | DVC tracks the dataset and all model artifacts |
| Pipeline as code | `dvc.yaml` defines every stage, its dependencies, and its outputs |
| Incremental execution | DVC only reruns stages whose inputs have changed |
| Experiment tracking | MLflow logs parameters, metrics, and model artifacts on every run |
| Model registry | Trained models are registered and versioned in the MLflow Model Registry |
| Configuration management | All paths and hyperparameters live in YAML files with no hardcoded values |
| Modular ML package | Source code is structured as an installable Python package with a clean src layout |
| Reproducibility | Any contributor can clone the repo and run `dvc repro` to get identical results |
| Containerisation | Dockerfile ensures the app runs consistently across all environments |
| REST API serving | Flask wraps the prediction pipeline and exposes it over HTTP |
| Automated deployment | GitHub Actions pushes to Hugging Face Spaces on every merge to main |
| Notebook-to-production | Each pipeline stage was prototyped in a Jupyter notebook before being productionised |

---

## Future Work

The current system is a solid, working foundation. These are the planned improvements for future versions:

### Model Improvements

- Extend to a multi-class classifier covering cysts, stones, and tumours in addition to normal scans
- Experiment with EfficientNetV2 and Vision Transformers for potentially higher accuracy
- Implement Grad-CAM visualisations in the web UI to show which regions of the scan influenced the prediction
- Increase training epochs and experiment with learning rate schedules and early stopping

### MLOps Infrastructure

- Add a model monitoring layer to detect data drift and prediction degradation in production
- Implement automated retraining triggers when model performance drops below a defined threshold
- Add a full test suite covering unit tests for pipeline components and integration tests for the API
- Set up a staging environment on HF Spaces to validate model changes before pushing to production

### Application

- Add a confidence score to the prediction response and display it numerically alongside the confidence bar
- Support batch prediction: accept multiple scans in a single request
- Add an admin dashboard showing prediction history, volume, and accuracy trends
- Implement user authentication for the `/train` endpoint to prevent unauthorised retraining

### Deployment

- Explore deployment on a GPU-backed inference endpoint for sub-second response times on large batches
- Package the prediction service as a standalone REST API with OpenAPI documentation

---

## About the Author

**Paul Sentongo** is a data scientist and applied AI researcher with a Master's degree in Data Science. He is passionate about building machine learning systems that go beyond the notebook: reproducible, traceable, and deployable in the real world. His research interests include deep learning for medical imaging, MLOps infrastructure, and the practical challenges of making AI work reliably at scale.

Paul is currently open to research positions and industry roles where he can contribute to meaningful AI projects and grow alongside motivated teams.

- GitHub: [github.com/sentongo-web](https://github.com/sentongo-web)
- LinkedIn: [linkedin.com/in/paul-sentongo-885041284](https://www.linkedin.com/in/paul-sentongo-885041284/)
- Email: [sentongogray1992@gmail.com](mailto:sentongogray1992@gmail.com)
