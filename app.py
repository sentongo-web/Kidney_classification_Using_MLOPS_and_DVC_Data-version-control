import os
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cnnClassifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Resolve model path once — prefer .keras (Keras 3 native), fall back to .h5
_keras_path = os.path.join("artifacts", "training", "model.keras")
_h5_path    = os.path.join("artifacts", "training", "model.h5")
MODEL_PATH  = _keras_path if os.path.isfile(_keras_path) else _h5_path

# Load the model once at startup so every request reuses the same in-memory model
_MODEL = None
_MODEL_ERROR = None
try:
    from tensorflow.keras.models import load_model
    _MODEL = load_model(MODEL_PATH)
    print(f"[startup] Model loaded from {MODEL_PATH}")
except Exception as _e:
    _MODEL_ERROR = str(_e)
    print(f"[startup] WARNING: model failed to load — {_MODEL_ERROR}")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    ok = _MODEL is not None
    return jsonify({
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "model_path": os.path.abspath(MODEL_PATH),
        "error": _MODEL_ERROR,
    }), 200 if ok else 503


@app.route("/train", methods=["GET", "POST"])
def train():
    os.system("python main.py")
    return "Training completed successfully!"


@app.route("/predict", methods=["POST"])
def predict():
    if _MODEL is None:
        return jsonify({"error": f"Model not loaded: {_MODEL_ERROR}"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(filepath)
        pipeline = PredictionPipeline(filepath, model=_MODEL)
        result = pipeline.predict()
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
