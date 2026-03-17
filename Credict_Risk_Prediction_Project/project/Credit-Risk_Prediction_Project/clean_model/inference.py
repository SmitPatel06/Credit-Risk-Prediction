import joblib
import os
import numpy as np

def model_fn(model_dir):
    """Load model for inference"""
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    """Parse input request"""
    if request_content_type == "application/json":
        import json
        data = json.loads(request_body)
        return np.array(data["features"]).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    prediction = model.predict(input_data)
    return prediction.tolist()

def output_fn(prediction, response_content_type):
    """Format output response"""
    import json
    return json.dumps({"prediction": prediction})
