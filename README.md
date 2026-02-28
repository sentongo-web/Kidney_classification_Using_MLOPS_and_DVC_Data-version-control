# Kidney Disease Classification — End-to-End MLOps Project

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6?logo=dvc)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask)](https://flask.palletsprojects.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)](https://github.com/features/actions)

---

## Overview

A production-grade, end-to-end **MLOps** project that classifies kidney CT scan images to detect kidney diseases using a fine-tuned **VGG16** convolutional neural network. This project goes beyond just building a model — it demonstrates the full lifecycle of a machine learning system: from raw data ingestion and versioning through model training, evaluation, and deployment as a live web application.

Built as a portfolio project to showcase applied skills in **deep learning**, **data versioning**, **experiment tracking**, **pipeline orchestration**, and **CI/CD automation**.

---

## Problem Statement

Kidney disease is one of the leading causes of death globally, often going undetected until late stages. Early and accurate classification of CT scan images can significantly improve patient outcomes. This project builds an automated image classification system to distinguish between healthy kidneys and those affected by disease, reducing manual radiological workload and enabling faster diagnosis.

---

## Key Features

- **Deep Learning Model** — Transfer learning with VGG16, fine-tuned on kidney CT scan dataset
- **Reproducible Pipelines** — Full ML pipeline managed and versioned with DVC (data ingestion → base model → training → evaluation)
- **Experiment Tracking** — All runs, parameters, and metrics logged with MLflow for full reproducibility
- **Configuration-Driven** — Hyperparameters and paths managed via YAML config files; no hardcoded values
- **Modular Codebase** — Clean, production-style package structure (`components`, `pipeline`, `entity`, `config`, `utils`)
- **REST API & Web UI** — Flask web application exposing a prediction endpoint with a frontend interface
- **CI/CD Ready** — GitHub Actions workflow scaffolded for automated testing and deployment

---

## Tech Stack

| Category             | Tools / Libraries                              |
|----------------------|------------------------------------------------|
| Deep Learning        | TensorFlow, Keras (VGG16 transfer learning)    |
| Data Version Control | DVC                                            |
| Experiment Tracking  | MLflow                                         |
| Web Framework        | Flask, Flask-CORS                              |
| Data Processing      | NumPy, Pandas, scikit-learn, SciPy             |
| Visualization        | Matplotlib, Seaborn                            |
| Configuration        | PyYAML, python-box, ensure                     |
| Packaging            | setuptools (`src` layout)                      |
| CI/CD                | GitHub Actions                                 |
| Environment          | Conda, pip                                     |

---

## Project Architecture

```
Kidney_classification_Using_MLOPS_and_DVC/
│
├── .github/workflows/          # CI/CD pipeline (GitHub Actions)
├── config/
│   └── config.yaml             # Centralized path & artifact configuration
├── params.yaml                 # Model hyperparameters (learning rate, epochs, etc.)
├── dvc.yaml                    # DVC pipeline stage definitions
│
├── src/cnnClassifier/
│   ├── components/             # Core ML logic (data ingestion, model, training, eval)
│   ├── pipeline/               # Stage-by-stage pipeline orchestration scripts
│   ├── entity/                 # Typed config dataclasses
│   ├── config/                 # Configuration manager
│   ├── utils/                  # Shared utilities (file I/O, logging)
│   └── constants/              # Project-wide constants
│
├── research/
│   └── trials.ipynb            # Exploratory notebooks for prototyping
├── templates/
│   └── index.html              # Frontend for the web prediction app
├── setup.py                    # Installable package definition
└── requirements.txt            # Python dependencies
```

---

## ML Pipeline (DVC Stages)

```
Data Ingestion  →  Base Model Preparation  →  Model Training  →  Model Evaluation
```

Each stage is defined in `dvc.yaml` and tracked by DVC, ensuring:
- Full pipeline reproducibility
- Incremental re-runs (only re-executes changed stages)
- Data and model artifact versioning

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.git
cd Kidney_classification_Using_MLOPS_and_DVC_Data-version-control
```

### 2. Create and Activate Conda Environment

```bash
conda create -n kidney python=3.13.9 -y
conda activate kidney
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Full DVC Pipeline

```bash
dvc repro
```

This executes all pipeline stages in order: data ingestion, base model setup, training, and evaluation.

### 5. Track Experiments with MLflow

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser to view all experiment runs, metrics, and parameters.

### 6. Launch the Web Application

```bash
python app.py
```

Open `http://localhost:8080` to use the prediction interface — upload a kidney CT scan image and get an instant classification result.

---

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | TBD    |
| Loss      | TBD    |

> Metrics are logged automatically to MLflow after each training run via `dvc repro`.

---

## MLOps Concepts Demonstrated

| Concept                        | Implementation                              |
|--------------------------------|---------------------------------------------|
| Data versioning                | DVC tracks datasets and model artifacts     |
| Pipeline as code               | `dvc.yaml` defines all pipeline stages      |
| Experiment tracking            | MLflow logs params, metrics, and artifacts  |
| Configuration management       | YAML-driven, no hardcoded values            |
| Modular ML package             | `src` layout, installable via `setup.py`    |
| Reproducibility                | `dvc repro` re-runs only stale stages       |
| CI/CD automation               | GitHub Actions workflow                     |
| REST API deployment            | Flask app serving model predictions         |

---

## Author

**Paul Sentongo**
- GitHub: [@sentongo-web](https://github.com/sentongo-web)
- Email: sentongogray1992@gmail.com
- Email: paulsentongo@eclipso.de