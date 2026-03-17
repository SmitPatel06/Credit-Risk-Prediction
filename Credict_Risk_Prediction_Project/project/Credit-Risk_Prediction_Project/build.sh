#!/bin/bash
set -e

# Train model
python train.py

# Package model and inference script
tar -czf model-credit-risk-clean.tar.gz model.joblib inference.py

# Verify package
echo "📦 Contents of model artifact:"
tar -tzf model-credit-risk-clean.tar.gz
