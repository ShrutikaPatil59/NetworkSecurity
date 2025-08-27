import os
import sys
from flask import Flask
from flask import request, jsonify, render_template
import joblib
import numpy as np
import certifi
ca=certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url=os.getenv("MONGO_URL_KEY")
print(mongo_db_url)

from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logger
from networksecurity.utils.ml_utils.model.estimator import NetworkSecurityModel


app = Flask(__name__)


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")   # Optional landing page


@app.route("/train", methods=["GET"])
def train():
    try:
        logger.info("Training pipeline triggered via API...")
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        return jsonify({
            "status": "success",
            "message": "Training pipeline completed successfully",
            "model_path": artifact.trained_model_file_path
        })
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expect JSON input

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Load latest trained model
        model_path = "artifacts/model/trained_model.pkl"
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found. Train the model first using /train"}), 404

        model: NetworkSecurityModel = joblib.load(model_path)

        # Convert input JSON to numpy array (expects flat feature list)
        input_array = np.array(data["features"]).reshape(1, -1)

        # Predict
        prediction = model.predict(input_array)
        prediction_class = int(prediction[0])

        return jsonify({
            "status": "success",
            "prediction": prediction_class
        })

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ---------- ENTRY ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
