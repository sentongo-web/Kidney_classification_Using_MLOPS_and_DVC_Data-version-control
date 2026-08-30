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
# KidneyDL: A Calibrated, Explainable Clinical Decision Support System for Kidney CT Classification

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![DVC](<https://img.shields.io/badge/DVC-Pipeline%20Versioning-945DD6?logo=dvc&logoColor=white>)](https://dvc.org/)
[![MLflow](<https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white>)](https://mlflow.org/)
[![DagsHub](<https://img.shields.io/badge/DagsHub-Remote%20Tracking-FF6B35?logoColor=white>)](https://dagshub.com/)
[![Flask](<https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white>)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![pytest](<https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white>)](https://docs.pytest.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Sentoz/kidney-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Author:** Paul Sentongo · Applied AI Researcher, Trustworthy Computer Vision & Digital Epidemiology
**Contact:** sentongopol@gmail.com · [GitHub](https://github.com/sentongo-web) · [LinkedIn](https://www.linkedin.com/in/paul-sentongo-885041284/)

---

## Abstract

Convolutional neural networks trained for radiological triage are frequently evaluated on a single axis — discriminative accuracy — while two properties that determine whether a clinician can actually *trust* the system are left unmeasured: whether the model's confidence scores are statistically honest (**calibration**), and whether its decisions are spatially and causally interpretable (**explainability**). This repository implements a reproducible MLOps pipeline for binary kidney CT scan classification (Normal vs. Tumor) built on a VGG16 transfer-learning backbone, and extends it along both of those axes: a vectorized **Expected Calibration Error (ECE)** estimator is computed on every evaluation run and logged alongside loss and accuracy, and an embedded **Explainable AI (XAI) engine** produces two complementary forms of post-hoc visual evidence for every positive finding — **Grad-CAM** spatial attention maps and **gradient-optimized counterfactual** reconstructions. The system is fully pipeline-as-code (DVC), experiment-tracked (MLflow/DagsHub), covered by an automated test suite (pytest, wired into CI), containerised (Docker), and served through a Flask web application deployed to Hugging Face Spaces.

**Keywords:** medical imaging, transfer learning, model calibration, expected calibration error, explainable AI, Grad-CAM, counterfactual explanation, MLOps, DVC, MLflow.

---

## 1. Motivation and Clinical Problem

Kidney disease is among the leading causes of death globally, and renal tumors often go undetected until later stages when treatment options are severely limited. Radiologists reviewing CT scans operate under significant time pressure, and an automated triage tool has genuine clinical value — but only if its outputs can be trusted and interrogated rather than accepted or dismissed as a black box.

This creates three concrete research requirements for a system positioned as *decision support* rather than a novelty classifier:

1. **Discriminative performance** — the model must separate Normal and Tumor scans reliably.
2. **Calibrated uncertainty** — a reported "97% confidence" must correspond to approximately 97% empirical correctness across the validation population, or the score is not actionable by a clinician.
3. **Interrogable decisions** — a positive finding must be accompanied by evidence a human can independently evaluate: *where* in the image the model is looking, and *what* the boundary between "Normal" and "Tumor" looks like in pixel space for this specific patient.

This project treats all three as first-class, pipeline-level artifacts rather than optional add-ons — each is computed and versioned by DVC on every run, not produced ad hoc in a notebook.

---

## 2. System Overview

```text
                    ┌─────────────────┐
   CT Scan Image ─▶ │  Flask REST API  │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │  PredictionPipeline    │
                  │  (confidence-gated)    │
                  └──────────┬─────────────┘
                             │
              ┌──────────────┴───────────────┐
              │                               │
        confidence < 0.80              confidence ≥ 0.80
              │                               │
              ▼                               ▼
      "InvalidImage"                 argmax(softmax) ∈ {Normal, Tumor}
      (reject: likely not                     │
       a kidney CT scan)          ┌────────────┴────────────┐
                                  │                          │
                              Normal                      Tumor
                                  │                          │
                             return label        ┌───────────┴───────────┐
                                                  │                       │
                                          ExplainabilityEngine   ExplainabilityEngine
                                          .generate_gradcam()    .generate_counterfactual()
                                                  │                       │
                                          Grad-CAM overlay        Counterfactual PNG
                                          (spatial evidence)     (pixel-space evidence)
```

Every arrow in this diagram is a DVC-tracked or MLflow-logged artifact, not an implicit side effect — see Section 4.

---

## 3. Methodology

### 3.1 Model Architecture

The classifier is a VGG16 backbone (Simonyan & Zisserman, 2014), pre-trained on ImageNet, with its original classification head removed. The frozen convolutional base — 13 layers of stacked 3×3 convolutions, terminating in `block5_conv3` → `block5_pool` — is retained for its strong low- and mid-level texture priors, which transfer well to the fine-grained tissue discrimination required in medical imaging. A new head (global spatial reduction → dense softmax over 2 classes) is trained on the kidney CT scan corpus while the base remains frozen, which lets the model reach strong performance without requiring a large labelled medical dataset.

`block5_conv3` — the terminal, deepest spatial feature representation in the backbone — is the layer this project targets for Grad-CAM (Section 3.3), since it is the last point in the network where spatial resolution and semantic abstraction are simultaneously high enough to produce a meaningful localization map.

### 3.2 Probability Calibration

A softmax score is not, by construction, a calibrated probability — a network can be systematically over- or under-confident while still being highly accurate. This project quantifies that gap with the **Expected Calibration Error (ECE)** (Guo et al., 2017; Naeini et al., 2015):

```text
ECE = Σ_{b=1}^{B}  ( |D_b| / N )  ·  | acc(D_b) − conf(D_b) |
```

where the confidence interval `[0, 1]` is partitioned into `B = 10` uniformly spaced bins `D_1, …, D_B`; `acc(D_b)` is the empirical accuracy of samples whose top-class softmax confidence falls in bin `b`; `conf(D_b)` is the mean predicted confidence within that bin; and `N` is the total validation cohort size. A perfectly calibrated model has `ECE = 0`: within every confidence bucket, stated confidence equals observed correctness.

The estimator (`compute_expected_calibration_error` in [`model_evaluation_mlflow.py`](src/cnnClassifier/components/model_evaluation_mlflow.py)) is a vectorized NumPy routine that runs a full inference sweep over the held-out validation generator, bins the resulting `(confidence, correctness)` pairs, and accumulates the weighted absolute gap. It is invoked automatically as part of the evaluation pipeline stage and its result is written to `scores.json` and logged as an MLflow metric on every run — it is not a one-off analysis script.

### 3.3 Explainable AI Layer

Two complementary, non-overlapping forms of post-hoc evidence are generated for every scan the model classifies as **Tumor** (a Normal reading has no pathological region to localize or counterfact against, so evidence generation is skipped for that branch by design):

**Grad-CAM — "where is the evidence?"**
Following Selvaraju et al. (2017), the gradient of the top predicted class score is backpropagated to the activations of `block5_conv3`, channel-averaged into per-channel importance weights, and used to produce a weighted sum of the feature maps. A ReLU filter retains only the *positive* spatial evidence (features that increase confidence in the predicted class, discarding those that argue against it), the result is min-max normalized, resized to the source resolution, colorized with a JET colormap, and alpha-blended over the raw scan. This answers: *which regions of renal tissue drove this classification?*

**Gradient-Optimized Counterfactual — "what would make it different?"**
Following the counterfactual-explanation framework of Wachter et al. (2017), the network's parameters are frozen and the input pixel tensor itself is treated as the only trainable variable. An SGD optimizer performs gradient descent for `steps` iterations on the negative log-likelihood of the target ("Normal") class:

```text
L(x) = − log p_θ(y = Normal | x),      x ← clip( x − lr · ∇ₓL(x), 0, 1 )
```

with `θ` (the network weights) held fixed throughout. The resulting image is not a diagnosis — it is a mathematical illustration of the minimal pixel-density reconfiguration the model's own decision surface associates with a healthy reading, letting a clinician visually compare "what the model saw" against "what would have changed its mind."

Both routines are implemented in [`explainability_engine.py`](src/cnnClassifier/components/explainability_engine.py) as `ExplainabilityEngine.generate_gradcam` and `ExplainabilityEngine.generate_counterfactual`, and are exercised end-to-end by the automated test suite (Section 7) and by the live `/predict` endpoint (Section 8).

---

## 4. Pipeline Architecture

The pipeline has five stages, each declared in [`dvc.yaml`](dvc.yaml) with explicit dependencies and outputs, so DVC only reruns what has actually changed.

```text
Stage 1        Stage 2        Stage 3       Stage 4          Stage 5
Data      →    Base Model →   Model    →    Model       →    Explainability
Ingestion      Preparation    Training      Evaluation        (Grad-CAM +
                                             (+ ECE)           Counterfactuals)
```

| Stage                     | Component                      | Output                                                       |
| ------------------------- | ------------------------------ | ------------------------------------------------------------ |
| 1. Data Ingestion         | `data_ingestion.py`          | `artifacts/data_ingestion/` (raw CT scan corpus)           |
| 2. Base Model Preparation | `prepare_base_model.py`      | Frozen VGG16 + custom softmax head                           |
| 3. Model Training         | `model_trainer.py`           | `artifacts/training/model.h5`                              |
| 4. Model Evaluation       | `model_evaluation_mlflow.py` | `scores.json` (loss, accuracy, **ECE**), MLflow run  |
| 5. Explainability         | `stage_05_explainability.py` | `artifacts/explainability/{gradcam,counterfactuals}/*.png` |

Stage 5 loads the registered model artifact, samples representative images from each class subfolder of the validation set, and regenerates the full visual-evidence tree on every sweep — the same `ExplainabilityEngine` used here is reused, unmodified, by the live inference path in `app.py`.

---

## 5. Model Performance

| Metric                           | Value                                                                |
| -------------------------------- | -------------------------------------------------------------------- |
| Accuracy                         | 89.9% (most recent registered run)                                   |
| Loss                             | 1.26                                                                 |
| Expected Calibration Error (ECE) | Computed automatically on every evaluation run — see`scores.json` |
| Architecture                     | VGG16 (fine-tuned)                                                   |
| Model Version                    | Registered in the MLflow Model Registry as`VGG16Model`             |
| Training Epochs                  | 5                                                                    |
| Optimiser                        | SGD (learning rate 0.01)                                             |
| Input Size                       | 224 × 224 × 3                                                      |
| Classes                          | Normal, Tumor                                                        |

The ECE row is intentionally reported as "computed automatically" rather than a fixed historical number: calibration is now a first-class metric produced by `dvc repro`, and the authoritative value for any given model version lives in that run's `scores.json` and MLflow record, not in this document. Re-running the pipeline (`dvc repro`) regenerates it.

All experiment runs, parameters, and model artifacts are tracked remotely on DagsHub:
[https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow](https://dagshub.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.mlflow)

---

## 6. Repository Structure

```text
Kidney_classification_Using_MLOPS_and_DVC/
│
├── .github/workflows/
│   ├── ci.yml                        Lint, install, run pytest suite, package sanity check
│   └── cd.yml                        Deploys to Hugging Face Spaces on push to main
│
├── config/config.yaml                Central path and artifact configuration (incl. explainability)
├── params.yaml                       All model hyperparameters
├── dvc.yaml / dvc.lock                Five-stage DVC pipeline definition and lock state
├── main.py                           Runs all five pipeline stages sequentially
├── app.py                            Flask web application (prediction + XAI evidence API)
├── Dockerfile                        Container definition for the prediction server
├── deploy_to_hf.py                   One-command deployment script for Hugging Face Spaces
├── requirements.txt                  Python dependencies
├── setup.py                          Installable package definition (src layout)
├── scores.json                       Latest evaluation metrics (loss, accuracy, ECE)
│
├── src/cnnClassifier/
│   ├── __init__.py                   Logger setup
│   ├── constants/                    Project-wide constants (config file paths)
│   ├── entity/config_entity.py       Typed dataclasses for every pipeline stage config
│   ├── config/configuration.py       ConfigurationManager: reads YAML, builds typed configs
│   ├── utils/common.py               Shared utilities: YAML/JSON I/O, directory creation, base64
│   ├── components/
│   │   ├── data_ingestion.py             Downloads and extracts the dataset
│   │   ├── prepare_base_model.py         Loads VGG16, adds the classification head
│   │   ├── model_trainer.py              Trains the model with augmentation support
│   │   ├── model_evaluation_mlflow.py    Evaluates, computes ECE, logs to MLflow
│   │   └── explainability_engine.py      Grad-CAM + gradient counterfactual algorithms
│   └── pipeline/
│       ├── stage_01_data_ingestion.py
│       ├── stage_02_prepare_base_model.py
│       ├── stage_03_model_trainer.py
│       ├── stage_04_model_evaluation.py
│       ├── stage_05_explainability.py    Regenerates the XAI evidence tree
│       └── prediction.py                 Confidence-gated inference + on-demand XAI triggering
│
├── tests/                             pytest suite (calibration math, config manager,
│                                       XAI engine, prediction pipeline, Flask integration)
│
├── research/                          Stage-by-stage prototyping notebooks
│
└── templates/index.html               Web UI: dark/light mode, drag-and-drop, 3-pane XAI dashboard
```

---

## 7. Testing and Validation

The pipeline's core scientific and engineering claims are covered by an automated `pytest` suite in [`tests/`](tests/), wired into CI (`.github/workflows/ci.yml`) so it runs on every push and pull request:

| Test module                       | What it verifies                                                                                                                                                                                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_calibration.py`           | The ECE estimator against hand-computed bin arithmetic, the zero-ECE degenerate case, and boundedness in`[0, 1]`                                                                                                                                       |
| `test_config_manager.py`        | `ConfigurationManager.get_explainability_config()` returns a correctly typed `ExplainabilityConfig` and materialises its directory tree                                                                                                              |
| `test_explainability_engine.py` | Grad-CAM produces a correctly shaped overlay and raises cleanly on a missing image; the counterfactual optimizer measurably increases the target class's probability after gradient descent                                                              |
| `test_prediction_pipeline.py`   | All three`PredictionPipeline` branches (Normal, Tumor, low-confidence `InvalidImage`), and that a failure inside the XAI engine is caught and never prevents a classification from being returned                                                    |
| `test_app.py`                   | The live Flask`/predict` route end-to-end via `app.test_client()` — missing file, unavailable model, a Normal upload (no XAI payload), a Tumor upload (base64 Grad-CAM + counterfactual returned), and resilience when the XAI engine itself throws |

Because a full-size VGG16 checkpoint is not committed to source control (it is a DVC/MLflow-tracked artifact), the test suite uses small deterministic Keras models built with a real layer named `block5_conv3`, so the exact `ExplainabilityConfig.target_layer_name` and gradient-computation code paths used in production are exercised unmodified — no test-only branches exist in `explainability_engine.py`.

Run the suite locally with:

```bash
pytest tests/ -v
```

---

## 8. The Web Application

The Flask app exposes:

| Route        | Method   | Description                                                          |
| ------------ | -------- | -------------------------------------------------------------------- |
| `/`        | GET      | Serves the prediction web UI                                         |
| `/predict` | POST     | Accepts an image file, returns classification + XAI evidence as JSON |
| `/health`  | GET      | Reports model-load status for readiness checks                       |
| `/train`   | GET/POST | Reruns`main.py` to retrain the model from scratch                  |

`/predict` response schema:

```json
[{
  "image": "Tumor",
  "confidence": 0.9731,
  "gradcam": "<base64-encoded PNG>",
  "counterfactual": "<base64-encoded PNG>"
}]
```

For a `"Normal"` verdict, `gradcam` and `counterfactual` are empty strings (no pathological region exists to explain). Predictions below a 0.80 softmax confidence threshold are returned as `{"image": "InvalidImage", ...}` rather than a forced Normal/Tumor call — a deliberate refusal behaviour for inputs that do not resemble the training distribution (e.g. a non-CT-scan upload).

The UI (`templates/index.html`) renders a live drag-and-drop scan preview, a confidence bar, and — for Tumor findings only — a three-pane research dashboard showing the original scan, the Grad-CAM attention overlay, and the counterfactual simulation side by side, in both light and dark mode.

---

## 9. Reproducing the Pipeline

### 9.1 Clone and install

```bash
git clone https://github.com/sentongo-web/Kidney_classification_Using_MLOPS_and_DVC_Data-version-control.git
cd Kidney_classification_Using_MLOPS_and_DVC_Data-version-control
conda create -n kidney python=3.10 -y
conda activate kidney
pip install -r requirements.txt
pip install -e .
```

### 9.2 Configure experiment tracking credentials

Create a `.env` file in the project root (gitignored, never committed):

```env
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 9.3 Run the full five-stage pipeline

```bash
dvc repro
```

DVC executes data ingestion, base model preparation, training, evaluation (with ECE), and explainability generation, in order, skipping any stage whose inputs are unchanged. After completion, `scores.json` holds the latest loss/accuracy/ECE, and `artifacts/explainability/` holds the regenerated Grad-CAM and counterfactual evidence tree.

### 9.4 Run the test suite

```bash
pytest tests/ -v
```

### 9.5 Launch the web application

```bash
python app.py
```

Open `http://localhost:7860`, upload a kidney CT scan, and receive a classification with confidence and (for Tumor findings) full XAI evidence.

### 9.6 Run with Docker

```bash
docker build -t kidney-classifier .
docker run -p 7860:7860 kidney-classifier
```

### 9.7 View experiment runs

```bash
mlflow ui
```

or visit the DagsHub MLflow URL in Section 5 for all remotely tracked runs.

---

## 10. CI/CD

Every push triggers the CI workflow (`.github/workflows/ci.yml`): dependency installation, an editable package install, the full `pytest` suite, and a source-structure sanity check. Every push to `main` additionally triggers the CD workflow (`.github/workflows/cd.yml`), which pushes the repository to Hugging Face Spaces; Spaces then detects the `Dockerfile` and rebuilds the deployed container automatically.

Because the trained model artifact is gitignored (DVC/MLflow-tracked, not committed), the initial deployment or any deployment that must include a newly trained model uses:

```bash
python deploy_to_hf.py
```

which uploads all necessary files, including the model artifact, via the Hugging Face Hub Python API.

---

## 11. Live Demo

**[https://huggingface.co/spaces/Sentoz/kidney-classifier](https://huggingface.co/spaces/Sentoz/kidney-classifier)**

Upload a kidney CT scan and receive a classification, a confidence score, and — for Tumor findings — the full Grad-CAM and counterfactual evidence panel, directly in your browser.

---

## 12. Limitations

- **Binary scope.** The model distinguishes Normal from Tumor only; it does not yet cover cysts, stones, or other renal pathologies that a full differential would require.
- **Dataset size and epochs.** Five training epochs on a single public CT scan corpus is sufficient to demonstrate the pipeline but is not a claim of clinical-grade generalisation across scanners, protocols, or populations.
- **Calibration is measured, not yet enforced.** ECE is computed and logged on every run, but the training objective itself is standard cross-entropy — no temperature scaling or calibration-aware loss is currently applied. A high ECE on a given run is a diagnostic signal, not (yet) something the pipeline automatically corrects.
- **Counterfactuals are illustrative, not generative-model-grade.** Because pixels are optimized directly against a frozen classifier rather than through a learned generative prior, counterfactual images can contain adversarial-looking artifacts rather than photorealistic "healthy tissue" — they should be read as a gradient map of the decision boundary, not a synthetic diagnosis.
- **Not a certified medical device.** This system is a research and portfolio artifact. It must never be used as a substitute for professional radiological or clinical judgment.

---

## 13. Future Work

- Multi-class extension covering cysts and stones in addition to Normal/Tumor
- Calibration-aware training (temperature scaling, focal loss, or Platt scaling) informed directly by the ECE signal this pipeline already produces
- Counterfactual generation through a learned generative prior (e.g. a diffusion or GAN-based inpainting model) for more clinically legible "healthy tissue" reconstructions
- Data-drift and prediction-drift monitoring in production, with automated retraining triggers
- OpenAPI documentation and a standalone REST API packaging of the prediction service

---

## 14. References

- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* arXiv:1409.1556.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI.
- Wachter, S., Mittelstadt, B., & Russell, C. (2017). *Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR.* Harvard Journal of Law & Technology.

---

## 15. About the Author

**Paul Sentongo** is a data scientist and applied AI researcher with a Master's degree in Data Science, specialising in trustworthy computer vision and digital epidemiology for resource-constrained clinical environments. His research interests span deep learning for medical imaging, model calibration and uncertainty quantification, explainable AI, and the practical MLOps infrastructure required to make such systems reliable, reproducible, and deployable at scale.

Paul is currently open to research positions and industry roles where he can contribute to meaningful AI work and grow alongside motivated teams.

- GitHub: [github.com/sentongo-web](https://github.com/sentongo-web)
- LinkedIn: [linkedin.com/in/paul-sentongo-885041284](https://www.linkedin.com/in/paul-sentongo-885041284/)
- Email: [sentongopol@gmail.com](mailto:sentongopol@gmail.com)
