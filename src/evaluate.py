
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from preprocessing import load_data, split_features_target

MODEL_PATH = "models/faircredit_model.joblib"


def main():
    # Load the dataset and separate features from target
    df = load_data()
    X, y = split_features_target(df)

    # Recreate the same test split used during model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Load the previously trained pipeline
    model = joblib.load(MODEL_PATH)

    # Generate test predictions and default probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Display evaluation metrics
    print("\nMODEL EVALUATION")
    print("=" * 45)
    print(f"Test records: {len(X_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(
        f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}"
    )
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
