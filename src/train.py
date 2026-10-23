
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Load the dataset
df = pd.read_excel(
    "data/default of credit card clients.xls",
    header=1
)

# 2. Rename the target column
df = df.rename(
    columns={"default payment next month": "DEFAULT"}
)

# 3. Separate features and target
X = df.drop(columns=["DEFAULT", "ID"])
y = df["DEFAULT"].astype(int)

# 4. Select categorical columns
categorical_cols = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

# 5. Select numerical columns
numerical_cols = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Create the preprocessor
preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_cols
    ),
    ("numerical", StandardScaler(), numerical_cols)
])

# 8. Create the model pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# 9. Train the model
model.fit(X_train, y_train)

# 10. Evaluate on the held-out test set
probabilities = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, probabilities)

print("Model: Logistic Regression")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Test ROC-AUC:", round(roc_auc, 3))

# 11. Save the complete pipeline
joblib.dump(model, "models/faircredit_model.joblib")

print("Model saved successfully!")
