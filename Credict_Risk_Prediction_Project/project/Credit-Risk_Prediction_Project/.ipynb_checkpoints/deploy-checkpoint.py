from sagemaker.sklearn.model import SKLearnModel
import sagemaker
import boto3
import os

# Create SageMaker session
session = sagemaker.Session()

# Get IAM role for execution
role = sagemaker.get_execution_role()

print("✅ Session and role ready")

# Ensure model.tar.gz exists
if not os.path.exists("model.tar.gz"):
    os.system("tar -czvf model.tar.gz -C model .")
    print("✅ Model.tar.gz created")

# Upload to S3
model_artifact = session.upload_data("model.tar.gz", key_prefix="credit-risk-model")
print("✅ Model uploaded to S3:", model_artifact)

# Create SageMaker model
model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point="inference.py",   # Ensure this is inside model/
    framework_version="1.2-1",    # Match sklearn version
    py_version="py3",
    sagemaker_session=session,
)

# Deploy endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print("✅ Endpoint deployed:", predictor.endpoint_name)

# Save endpoint name to file
with open("endpoint.txt", "w") as f:
    f.write(predictor.endpoint_name)
print("✅ Endpoint name saved to endpoint.txt")
