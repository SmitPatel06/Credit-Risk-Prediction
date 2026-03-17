import joblib
import numpy as np

def model_fn(model_dir):
    model = joblib.load(f"{model_dir}/model.joblib")
    return model

def input_fn(request_body, request_content_type):
    import json
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["features"]).reshape(1, -1)
    else:
        raise ValueError("Unsupported content type: " + request_content_type)

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)  # add probability
    return {"prediction": prediction.tolist(), "probabilities": probabilities.tolist()}

def output_fn(prediction, accept):
    import json
    return json.dumps(prediction), accept
