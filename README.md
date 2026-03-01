# Kidney Disease Classification — End-to-End MLOps Project

[![CI](https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control/actions/workflows/ci.yml/badge.svg)](https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control/actions/workflows/ci.yml)
[![CD](https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control/actions/workflows/cd.yml/badge.svg)](https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6?logo=dvc)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20ECR-FF9900?logo=amazonaws)](https://aws.amazon.com/)

---

## Overview

A production-grade, end-to-end **MLOps** project that classifies kidney CT scan images to detect kidney disease using a fine-tuned **VGG16** convolutional neural network. This project goes beyond just building a model — it demonstrates the complete lifecycle of a machine learning system: from raw data ingestion and versioning through model training, evaluation, containerisation, and automated deployment to the cloud.

Built as a portfolio project to showcase applied skills in **deep learning**, **data versioning**, **experiment tracking**, **pipeline orchestration**, **Docker**, and **CI/CD automation with GitHub Actions**.

---

## Problem Statement

Kidney disease is one of the leading causes of death globally, often going undetected until late stages. Early and accurate classification of CT scan images can significantly improve patient outcomes. This project builds an automated image classification system to distinguish between healthy kidneys and those affected by disease, reducing manual radiological workload and enabling faster diagnosis.

---

## Key Features

- **Deep Learning Model** — Transfer learning with VGG16, fine-tuned on a kidney CT scan dataset
- **Reproducible ML Pipelines** — Full pipeline versioned with DVC: data ingestion → base model → training → evaluation
- **Experiment Tracking** — All runs, parameters, and metrics logged with MLflow for full auditability
- **Configuration-Driven** — Hyperparameters and paths managed via YAML config files; zero hardcoded values
- **Modular Codebase** — Production-style Python package (`components`, `pipeline`, `entity`, `config`, `utils`)
- **REST API & Web UI** — Flask app exposing a `/predict` endpoint with a browser-based upload interface
- **Containerised** — Dockerfile for consistent, environment-independent builds
- **CI/CD Automated** — GitHub Actions runs integration checks on every push and deploys to AWS EC2 on merge to `main`

---

## Tech Stack

| Category              | Tools / Libraries                                  |
|-----------------------|----------------------------------------------------|
| Deep Learning         | TensorFlow, Keras (VGG16 transfer learning)        |
| Data Version Control  | DVC                                                |
| Experiment Tracking   | MLflow                                             |
| Web Framework         | Flask, Flask-CORS                                  |
| Data Processing       | NumPy, Pandas, scikit-learn, SciPy                 |
| Visualisation         | Matplotlib, Seaborn                                |
| Configuration         | PyYAML, python-box, ensure                         |
| Packaging             | setuptools (`src` layout, `pip install -e .`)      |
| Containerisation      | Docker                                             |
| Cloud Deployment      | AWS EC2 (runner) + AWS ECR (image registry)        |
| CI/CD                 | GitHub Actions                                     |
| Environment           | Conda / pip                                        |

---

## Project Architecture

```text
Kidney_classification_Using_MLOPS_and_DVC/
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # CI: lint & package check on every push / PR
│       └── cd.yml              # CD: Docker build → ECR push → EC2 deploy on main
│
├── config/
│   └── config.yaml             # Centralised path & artifact configuration
├── params.yaml                 # Model hyperparameters (LR, epochs, batch size, etc.)
├── dvc.yaml                    # DVC pipeline stage definitions
│
├── src/cnnClassifier/
│   ├── components/             # Core ML logic (data ingestion, model prep, training, eval)
│   ├── pipeline/               # Stage-by-stage orchestration scripts
│   ├── entity/                 # Typed config dataclasses
│   ├── config/                 # Configuration manager (reads config.yaml + params.yaml)
│   ├── utils/
│   │   └── common.py           # Shared utilities: YAML, JSON, joblib, base64, logging
│   └── constants/              # Project-wide constants (config paths)
│
├── research/
│   └── trials.ipynb            # Exploratory notebooks used to prototype each pipeline stage
├── templates/
│   └── index.html              # Frontend for the web prediction app
│
├── Dockerfile                  # Container definition for the Flask prediction server
├── main.py                     # Pipeline entry point — runs all DVC stages end-to-end
├── app.py                      # Flask web application (prediction API + UI)
├── setup.py                    # Installable package definition
└── requirements.txt            # Python dependencies
```

---

## ML Pipeline (DVC Stages)

```text
Data Ingestion  →  Base Model Preparation  →  Model Training  →  Model Evaluation
```

Each stage is defined in `dvc.yaml` and tracked by DVC, ensuring:

- Full pipeline reproducibility across machines
- Incremental re-runs — only stale stages are re-executed
- Data and model artifact versioning linked to Git commits

---

## CI/CD Pipeline (GitHub Actions)

### Continuous Integration — `ci.yml`

Triggered on **every push** to any branch and on **pull requests to `main`**.

| Step | Action |
| --- | --- |
| Checkout | Fetch the latest code |
| Setup Python 3.10 | Cached pip environment |
| Install dependencies | `pip install -r requirements.txt` |
| Package check | `pip install -e .` then import all core modules |
| Utility check | Verify all `common.py` functions import correctly |

### Continuous Deployment — `cd.yml`

Triggered on **push to `main`** (after CI passes).

| Step | Action |
| --- | --- |
| Build Docker image | `docker build` using the project Dockerfile |
| Push to AWS ECR | Tags image with commit SHA and `latest` |
| Deploy to AWS EC2 | Self-hosted runner pulls latest image and restarts the container |

#### Required GitHub Secrets

Set these in **Settings → Secrets and variables → Actions**:

| Secret | Description |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | IAM user access key with ECR + EC2 permissions |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret key |
| `AWS_ECR_LOGIN_URI` | ECR registry URI (e.g. `123456789.dkr.ecr.us-east-1.amazonaws.com`) |

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.git
cd Kidney_classification_Using_MLOPS_and_DVC_Data-version-control
```

### 2. Create and Activate Conda Environment

```bash
conda create -n kidney python=3.10 -y
conda activate kidney
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Run the Full DVC Pipeline

```bash
dvc repro
```

Executes all pipeline stages in order: data ingestion → base model setup → training → evaluation.

### 5. Track Experiments with MLflow

```bash
mlflow ui
```

Open `http://localhost:5000` to browse all experiment runs, compare metrics, and inspect parameters.

### 6. Launch the Flask Web Application

```bash
python app.py
```

Open `http://localhost:8080` — upload a kidney CT scan image and receive an instant classification result.

---

## Run with Docker

```bash
# Build the image
docker build -t kidney-classifier .

# Run the container
docker run -p 8080:8080 kidney-classifier
```

Open `http://localhost:8080` in your browser.

---

## Results

| Metric   | Value |
|----------|-------|
| Accuracy | TBD   |
| Loss     | TBD   |

> Metrics are logged automatically to MLflow after each `dvc repro` run.

---

## MLOps Concepts Demonstrated

| Concept                   | Implementation                                        |
|---------------------------|-------------------------------------------------------|
| Data versioning           | DVC tracks datasets and model artifacts               |
| Pipeline as code          | `dvc.yaml` defines and connects all pipeline stages   |
| Experiment tracking       | MLflow logs params, metrics, and model artifacts      |
| Configuration management  | YAML-driven; no hardcoded values anywhere             |
| Modular ML package        | `src` layout, installable via `setup.py`              |
| Reproducibility           | `dvc repro` re-runs only what changed                 |
| Containerisation          | Dockerfile for consistent runtime environments        |
| CI automation             | GitHub Actions verifies every push                    |
| CD automation             | GitHub Actions deploys to AWS EC2 on every main merge |
| REST API deployment       | Flask serves predictions over HTTP                    |

---

## Author

**Paul Sentongo**

- GitHub: [@sentongo-web](https://github.com/sentongo-web)
- Email: sentongogray1992@gmail.com
- Email: paulsentongo@eclipso.de
