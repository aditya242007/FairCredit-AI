
import pandas as pd


DATA_PATH = "data/default of credit card clients.xls"


def load_data(path=DATA_PATH):
    """Load the credit card default dataset."""
    df = pd.read_excel(path, header=1)

    df = df.rename(
        columns={"default payment next month": "DEFAULT"}
    )

    return df


def split_features_target(df):
    """Separate model features from the target."""
    X = df.drop(columns=["DEFAULT", "ID"])
    y = df["DEFAULT"].astype(int)

    return X, y
