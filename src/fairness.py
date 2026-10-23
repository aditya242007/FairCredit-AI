
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from preprocessing import load_data, split_features_target

MODEL_PATH = "models/faircredit_model.joblib"


def calculate_group_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Support": len(y_true),
        "Predicted Default Rate": y_pred.mean(),
        "Recall": tp / (tp + fn) if (tp + fn) else 0,
        "False Positive Rate": fp / (fp + tn) if (fp + tn) else 0,
    }


def main():
    df = load_data()
    X, y = split_features_target(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    results = pd.DataFrame({
        "SEX": X_test["SEX"],
        "AGE": X_test["AGE"],
        "Actual": y_test,
        "Predicted": y_pred,
    }, index=X_test.index)

    print("\nFAIRNESS ANALYSIS")
    print("=" * 60)

    print("\nMetrics by SEX")
    sex_results = []

    for group, data in results.groupby("SEX"):
        metrics = calculate_group_metrics(
            data["Actual"].to_numpy(),
            data["Predicted"].to_numpy(),
        )
        metrics["Group"] = group
        sex_results.append(metrics)

    print(pd.DataFrame(sex_results).set_index("Group").round(4))

    results["AGE_GROUP"] = pd.cut(
        results["AGE"],
        bins=[0, 29, 39, 49, 59, 120],
        labels=["29 or younger", "30-39", "40-49", "50-59", "60+"],
        include_lowest=True,
    )

    print("\nMetrics by AGE GROUP")
    age_results = []

    for group, data in results.groupby("AGE_GROUP", observed=True):
        metrics = calculate_group_metrics(
            data["Actual"].to_numpy(),
            data["Predicted"].to_numpy(),
        )
        metrics["Group"] = group
        age_results.append(metrics)

    print(pd.DataFrame(age_results).set_index("Group").round(4))


if __name__ == "__main__":
    main()
