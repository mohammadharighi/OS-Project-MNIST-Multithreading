"""
app.py

Flask server providing the web UI and the following endpoints:
    GET  /         -> Main page (index.html)
    POST /train    -> Start training with parameters: epoch, concurrent_epochs, batch_size
    GET  /status   -> Real-time training status (for polling)
    POST /predict  -> Predict the digit of an uploaded image

Author: Mohammad Harighi
"""

import io
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify

from model.cnn import CNNTrainer

app = Flask(__name__)

# Singleton instance of CNNTrainer that holds the model and training state
trainer = CNNTrainer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(force=True, silent=True) or {}

    try:
        epochs = int(data.get("epochs", 5))
        concurrent_epochs = int(data.get("concurrent_epochs", 2))
        batch_size = int(data.get("batch_size", 128))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid input parameters."}), 400

    epochs = max(1, min(epochs, 50))
    concurrent_epochs = max(1, min(concurrent_epochs, epochs))
    batch_size = max(16, min(batch_size, 2048))

    started = trainer.start_training_async(epochs, concurrent_epochs, batch_size)
    if not started:
        return jsonify({"ok": False, "error": "Another training process is already running."}), 409

    return jsonify({
        "ok": True,
        "message": "Training started.",
        "epochs": epochs,
        "concurrent_epochs": concurrent_epochs,
        "batch_size": batch_size,
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify(trainer.get_status())


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image was sent."}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("L")  # Grayscale
        img = img.resize((28, 28))

        arr = np.array(img).astype("float32")

        # If the image has a light background and dark digit (e.g., a photo on white paper),
        # invert it to match MNIST format (bright digit, dark background).
        if arr.mean() > 127:
            arr = 255.0 - arr

        arr = arr / 255.0

        digit, confidence, probs = trainer.predict(arr)

        return jsonify({
            "ok": True,
            "digit": digit,
            "confidence": round(confidence * 100, 2),
            "probabilities": probs,
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    # debug=False and threaded=True because we manage multi-threading at the model level ourselves
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)