# Credit-Risk-Prediction
This project is a Credit Risk Prediction System that uses AWS SageMaker for machine learning inference and Streamlit as the frontend interface. Users can input applicant details such as Age, Annual Income, and Loan Amount to predict the probability of loan default.

1. Overview:
   
This project is a Credit Risk Prediction System that uses AWS SageMaker for machine
learning inference and Streamlit as the frontend interface.
Users can input applicant details such as Age, Annual Income, and Loan Amount to predict
the probability of loan default.

2. Project Structure:

File Name Description:

build.sh Shell script to train, package, and verify the ML model before deployment.
credit_app.py Streamlit-based frontend app that interacts with the deployed AWS SageMaker
model.
Untitled.ipynb Jupyter Notebook (used for exploratory data analysis, model training, or
experiments).

3. Requirements

Python Version:
Python 3.8 or later
Python Libraries:
streamlit
boto3
json (built-in)
os (built-in)
Installation Command:
pip install streamlit boto3


4. How to Run the Project

Step 1: Train & Package the Model
Run the build.sh script to:
1. Train the ML model (train.py should be present in the same directory).
2. Package the trained model (model.joblib) and inference script (inference.py) into a Command to run:
chmod + build.sh
./build.sh
This will generate the file:
model-credit-risk-clean.tar.gz
Step 2: Deploy the Model on AWS SageMaker
• Option 1: If you already have a deployed SageMaker endpoint:
Save the endpoint name in a file named endpoint.txt in the project directory.
• Option 2: If you do not have an endpoint:
The application will automatically fetch the latest available SageMaker endpoint.
Note: Ensure AWS credentials are configured in ~/.aws/credentials
Step 3: Run the Streamlit Application
Command to start the app:
streamlit run credit_app.py

5. Using the Application

   1. Open the app in your browser using the link provided by Streamlit.
   2. Enter the following applicant details:
    o Age
    o Annual Income ($)
    o Loan Amount ($)
   3. Click Predict Risk to receive the result.
Outputs Displayed:
 • Prediction:
 o Default (1) – High risk of loan default.
 o No Default (0) – Low risk of loan default.
 • Probability values for both classes.
 • Progress bar visualizing default risk.
 • Color-coded risk message (red for high risk, green for low risk).

6. Model Packaging Process (build.sh):

The build.sh script performs the following steps:
1. Train the model (train.py).
2. Package the model and inference script into model-credit-risk-clean.tar.gz.
3. Verify the package contents by listing files inside the archive.


7. Notes
 
 • AWS CLI must be configured with the correct profile and permissions to use SageMaker.
 • The following files are required for the complete workflow but are not included here:
 o train.py (model training script)
 o inference.py (prediction script)
 • The Untitled.ipynb notebook can be used for dataset analysis, preprocessing, and experimenting with different ML models before deployment.
.tar.gz file.
 Verify the packaged contents.
