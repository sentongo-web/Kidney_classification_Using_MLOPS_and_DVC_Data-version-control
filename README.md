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

# KidneyDL: Calibrated and Explainable CT Classification for Decision Support

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline%20Versioning-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Remote%20Tracking-FF6B35?logoColor=white)](https://dagshub.com/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Sentoz/kidney-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Author:** Paul Sentongo  
**Contact:** sentongopol@gmail.com · [GitHub](https://github.com/sentongo-web) · [LinkedIn](https://www.linkedin.com/in/paul-sentongo-885041284/)

---

## Abstract

Convolutional neural networks built for radiological triage are often evaluated solely on classification accuracy. This overlooks two factors critical to clinical adoption: **probability calibration** (whether model confidence reflects true error rates) and **explainability** (whether spatial predictions are interpretable by clinicians). 

This project implements an end-to-end MLOps pipeline for binary kidney CT scan classification (Normal vs. Tumor) using a fine-tuned VGG16 backbone. The system extends basic classification in two key ways:
1. Calculates a vectorized **Expected Calibration Error (ECE)** score during model evaluation to monitor prediction confidence.
2. Integrates an **Explainable AI (XAI) engine** that generates post-hoc visual evidence for positive findings using **Grad-CAM** spatial maps and **gradient-optimized counterfactual** reconstructions.

The pipeline is version-controlled with DVC, tracked with MLflow/DagsHub, covered by an automated pytest suite, containerized with Docker, and served via a Flask web application on Hugging Face Spaces.

**Keywords:** medical imaging, transfer learning, model calibration, expected calibration error, explainable AI, Grad-CAM, counterfactual explanations, MLOps, DVC, MLflow.

---

## 1. Clinical Context and Motivation

Renal tumors often go undetected until advanced stages when treatment options become limited. While automated image analysis can assist radiologists under heavy workloads, deep learning models are rarely used in production if their outputs act as an uninterpretable black box.

For a classifier to function safely as decision support, it needs:

1. **Reliable separation** between Normal and Tumor scans.
2. **Calibrated confidence scores** so that a 95% confidence score aligns with historical accuracy across similar cases.
3. **Visual evidence** allowing radiologists to verify *where* the model is looking and *what* visual features drove the classification.

Rather than running these evaluations manually, this pipeline tracks accuracy, calibration metrics, and visual explanations automatically during each build run using DVC.

---

## 2. System Architecture

```text
                    ┌─────────────────┐
   CT Scan Image ─▶ │  Flask REST API │
                    └────────┬────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │   PredictionPipeline  │
                  │  (confidence gated)   │
                  └──────────┬────────────┘
                             │
              ┌──────────────┴───────────────┐
              │                              │
        confidence < 0.80              confidence ≥ 0.80
              │                              │
              ▼                              ▼
      "InvalidImage"                 argmax(softmax) ∈ {Normal, Tumor}
     (Rejects non-CT                         │
      or noisy inputs)            ┌──────────┴──────────┐
                                  │                     │
                               Normal                 Tumor
                                  │                     │
                             Return label     ┌─────────┴─────────┐
                                              │                   │
                                        Grad-CAM            Counterfactual
                                        Overlay             Reconstruction
                                    (Spatial Focus)      (Pixel Reconfiguration)
```

---

## 3. Technical Methodology

### 3.1 Model Architecture

The model uses a VGG16 backbone pre-trained on ImageNet with the original classification head removed. The frozen convolutional base (13 layers ending in `block5_conv3` → `block5_pool`) provides feature extraction tailored for texture patterns in medical scans. 

A custom classification head (global average pooling followed by a dense softmax layer) is trained on the kidney CT dataset. The `block5_conv3` layer is targeted for Grad-CAM generation because it retains spatial context alongside high-level feature representations.

### 3.2 Probability Calibration

Standard softmax outputs can produce overconfident predictions. To track how well predicted probabilities mirror actual accuracy, the evaluation stage calculates Expected Calibration Error (ECE):

$$ECE = \sum_{b=1}^{B} rac{|D_b|}{N} \Big| 	ext{acc}(D_b) - 	ext{conf}(D_b) \Big|$$

The confidence range $[0, 1]$ is split into $B = 10$ bins ($D_1 \dots D_B$). For each bin, the absolute difference between empirical accuracy ($	ext{acc}$) and average confidence ($	ext{conf}$) is weighted by the sample size ($N$). 

The ECE calculation is implemented as a vectorized NumPy function in `src/cnnClassifier/components/model_evaluation_mlflow.py` and runs automatically during pipeline execution.

### 3.3 Explainable AI (XAI) Engine

When the model detects a **Tumor**, the system generates two forms of visual evidence:

* **Grad-CAM (Spatial Focus):** Gradients from the target class score flow back to `block5_conv3` activations to compute channel weights. Applying a ReLU filter isolates features that positively support the tumor classification. The resulting heatmap is upsampled and overlaid onto the scan.
* **Counterfactual Explanations (Pixel Reconfiguration):** Using gradient descent, the input pixel values are iteratively modified while model weights remain frozen. The optimization minimizes the loss relative to the "Normal" class target. This yields an image showing the minimum pixel changes required for the model to reclassify the scan as healthy.

---

## 4. Pipeline Stages

The workflow is defined across five modular DVC stages in `dvc.yaml`:

| Stage | Script | Output / Artifact |
| :--- | :--- | :--- |
| **1. Data Ingestion** | `data_ingestion.py` | Extracted CT scan dataset |
| **2. Base Model Prep** | `prepare_base_model.py` | VGG16 structure + custom head |
| **3. Model Training** | `model_trainer.py` | `artifacts/training/model.h5` |
| **4. Evaluation** | `model_evaluation_mlflow.py` | Loss, Accuracy, ECE score in `scores.json` |
| **5. Explainability** | `stage_05_explainability.py` | XAI visual artifacts |

---

## 5. Model Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 89.9% |
| **Loss** | 1.26 |
| **Expected Calibration Error (ECE)** | Evaluated automatically (logged to `scores.json`) |
| **Architecture** | VGG16 (Transfer Learning) |
| **Optimizer** | SGD (learning rate = 0.01) |
| **Input Shape** | 224 × 224 × 3 |

Full experiment runs, parameters, and logged artifacts are accessible on DagsHub:  
[DagsHub MLflow Dashboard](https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow)

---

## 6. Repository Layout

```text
Kidney_classification_Using_MLOPS_and_DVC/
├── .github/workflows/       # CI/CD pipelines (Pytest and HF deployment)
├── config/config.yaml       # Path and pipeline configurations
├── params.yaml              # Hyperparameters and training settings
├── dvc.yaml                 # DVC pipeline stage specifications
├── main.py                  # Pipeline execution runner
├── app.py                   # Flask server and inference routes
├── Dockerfile               # Application container build spec
├── requirements.txt         # Project dependencies
├── setup.py                 # Package setup file
├── src/cnnClassifier/       # Source code modules
│   ├── components/          # Ingestion, training, evaluation, XAI modules
│   ├── pipeline/            # DVC execution wrappers & inference logic
│   └── utils/               # Common helper functions
├── tests/                   # Automated pytest suite
└── templates/index.html     # Web UI dashboard
```

---

## 7. Testing

Automated tests cover pipeline configurations, calibration logic, XAI output generation, and endpoint responses. Run tests locally using:

```bash
pytest tests/ -v
```

---

## 8. Web Application API

The Flask application (`app.py`) serves the web dashboard and prediction endpoints.

### API Routes

* `GET /` — Serves the frontend interface.
* `POST /predict` — Processes uploaded scans and returns classification data with base64-encoded visual explanations.
* `GET /health` — Simple status check endpoint.

#### Example `/predict` Response

```json
[
  {
    "image": "Tumor",
    "confidence": 0.9731,
    "gradcam": "<base64_encoded_png>",
    "counterfactual": "<base64_encoded_png>"
  }
]
```

*Note: Scans returning confidence scores below 0.80 are classified as `InvalidImage` to filter out non-CT inputs or out-of-distribution uploads.*

---

## 9. Local Setup & Reproduction

### 9.1 Installation

```bash
# Clone repository
git clone https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.git
cd Kidney_classification_Using_MLOPS_and_DVC_Data-version-control

# Create and activate environment
conda create -n kidney python=3.10 -y
conda activate kidney

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 9.2 Execution

1. **Set up tracking credentials** (Optional for DagsHub logging):
   Create a `.env` file containing:
   ```env
   MLFLOW_TRACKING_USERNAME=your_username
   MLFLOW_TRACKING_PASSWORD=your_token
   ```

2. **Run the pipeline:**
   ```bash
   dvc repro
   ```

3. **Start the Flask app:**
   ```bash
   python app.py
   ```
   Access the dashboard at `http://localhost:7860`.

4. **Run via Docker:**
   ```bash
   docker build -t kidney-classifier .
   docker run -p 7860:7860 kidney-classifier
   ```

---

## 10. Live Demo

The application is deployed on Hugging Face Spaces:  
**[Live Web Application](https://huggingface.co/spaces/Sentoz/kidney-classifier)**

---

## 11. Current Limitations

* **Scope:** The model performs binary classification (Normal vs. Tumor). It does not differentiate between cysts, stones, or secondary renal conditions.
* **Calibration post-processing:** ECE is calculated as an evaluation metric, but confidence post-processing (e.g., Platt scaling or temperature scaling) is not applied during inference.
* **Counterfactual representation:** Counterfactual images optimize raw pixel values against fixed model gradients. While effective for probing model boundaries, they are direct mathematical optimizations rather than photorealistic generative reconstructions.

---

## 12. References

* Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556.
* Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
* Guo, C., et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
* Wachter, S., et al. (2017). *Counterfactual Explanations without Opening the Black Box.* Harvard Journal of Law & Technology.

---

## 13. Author Information

**Paul Sentongo**  
Data Scientist & Applied AI Researcher  
Specializing in computer vision applications and MLOps infrastructure.

* **GitHub:** [github.com/sentongo-web](https://github.com/sentongo-web)
* **LinkedIn:** [linkedin.com/in/paul-sentongo-885041284](https://www.linkedin.com/in/paul-sentongo-885041284/)
* **Email:** sentongopol@gmail.com
