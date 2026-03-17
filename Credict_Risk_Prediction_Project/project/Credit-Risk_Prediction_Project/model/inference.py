import joblib
import os
import json
import pandas as pd
import pickle

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    print(f"🔍 Loading model from: {model_path}")
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded via joblib")
    except Exception as e:
        print(f"⚠️ joblib failed, retrying with pickle. Error: {e}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded via pickle")
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        print(f"📥 Input received: {data}")
        return pd.DataFrame([data])
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    print(f"🔮 Prediction: {prediction.tolist()}")
    return prediction.tolist()

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps({"prediction": prediction})
    raise ValueError(f"Unsupported content type: {response_content_type}")
