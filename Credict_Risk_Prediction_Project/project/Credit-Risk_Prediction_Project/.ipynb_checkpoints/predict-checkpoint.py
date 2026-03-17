import joblib
import pandas as pd

clf = joblib.load("model.joblib")

# Example input
sample = pd.DataFrame([{
    "person_age": 30,
    "person_income": 40000,
    "loan_amnt": 15000
}])

prediction = clf.predict(sample)
print("Prediction:", prediction[0])
