import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("processed_credit_risk.csv")

print("Columns in dataset:", df.columns.tolist())
print("First 5 rows:\n", df.head())

# Set the correct target column
target_col = "loan_status"

if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found in {df.columns.tolist()}")

print(f"✅ Using target column: {target_col}")

# Features & Target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "model.joblib")
print("✅ Model trained and saved as model.joblib")
