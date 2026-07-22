import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("student_data.csv")

# Target
data["Final_Result"] = data["G3"].apply(lambda x: 1 if x >= 10 else 0)

X = data.drop(["G3", "Final_Result"], axis=1)
y = data["Final_Result"]

# Columns
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train
pipeline.fit(X, y)

# Save
joblib.dump(pipeline, "student_pipeline.pkl")

print("✅ Model saved successfully")
