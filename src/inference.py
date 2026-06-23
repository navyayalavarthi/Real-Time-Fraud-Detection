
import joblib
import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    logger.info(f"Loading model from {model_dir}")
    logger.info(f"Files in model dir: {os.listdir(model_dir)}")

    model = joblib.load(os.path.join(model_dir, "fraud_model.pkl"))
    scaler_amount = joblib.load(os.path.join(model_dir, "scaler_amount.pkl"))
    scaler_time = joblib.load(os.path.join(model_dir, "scaler_time.pkl"))

    logger.info("Model loaded successfully.")
    return {
        "model": model,
        "scaler_amount": scaler_amount,
        "scaler_time": scaler_time
    }

def input_fn(request_body, content_type="application/json"):
    logger.info(f"Received request. Content type: {content_type}")
    if content_type == "application/json":
        data = json.loads(request_body)
        return data
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    scaler_amount = model_dict["scaler_amount"]
    scaler_time = model_dict["scaler_time"]

    data = input_data.copy()

    amount = float(data.pop("Amount", 0))
    time_val = float(data.pop("Time", 0))

    data["Amount_scaled"] = float(scaler_amount.transform([[amount]])[0][0])
    data["Time_scaled"] = float(scaler_time.transform([[time_val]])[0][0])
    data["hour"] = int(time_val / 3600) % 24
    data["log_amount"] = float(np.log1p(amount))

    feature_values = [float(v) for v in data.values()]
    features = np.array(feature_values).reshape(1, -1)

    prob = float(model.predict_proba(features)[0][1])
    logger.info(f"Prediction: {prob}")
    return prob

def output_fn(prediction, accept="application/json"):
    prob = float(prediction)

    if prob > 0.7:
        risk = "HIGH"
        action = "Block transaction immediately"
    elif prob > 0.4:
        risk = "MEDIUM"
        action = "Flag for manual review"
    else:
        risk = "LOW"
        action = "Allow transaction"

    result = {
        "fraud_probability": round(prob, 4),
        "is_fraud": bool(prob > 0.5),
        "risk_level": risk,
        "recommended_action": action
    }
    return json.dumps(result)
