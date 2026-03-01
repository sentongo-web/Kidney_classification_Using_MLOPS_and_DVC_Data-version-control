import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cnnClassifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    os.system("python main.py")
    return "Training completed successfully!"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    pipeline = PredictionPipeline(filepath)
    result = pipeline.predict()

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
