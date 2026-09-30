"""Train the baseline credit-card fraud detection model.

This script mirrors the core modelling workflow in the notebook:
- load the Parquet dataset
- handle missing values
- encode categorical variables
- split the data
- scale features
- apply SMOTE to the training data
- train XGBoost
- print the classification report

The raw dataset is intentionally not stored in the repository.
"""

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path("data/ieee_fraud_detection.parquet")
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load the transaction data from a Parquet file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )
    return pd.read_parquet(path, engine="pyarrow")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the baseline project's missing-value and encoding steps."""
    data = df.copy()

    # TransactionID is an identifier, not a predictive feature.
    if "TransactionID" in data.columns:
        data = data.set_index("TransactionID")

    # Numerical missing values are replaced with the feature median.
    numeric_columns = data.select_dtypes(include="number").columns
    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())

    # Categorical missing values are replaced with the most frequent category.
    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns
    for column in categorical_columns:
        mode = data[column].mode(dropna=True)
        if not mode.empty:
            data[column] = data[column].fillna(mode.iloc[0])

    # Preserve the original project's encoding strategy.
    if "P_emaildomain" in data.columns:
        encoder = LabelEncoder()
        data["P_emaildomain"] = encoder.fit_transform(
            data["P_emaildomain"].astype(str)
        )

    one_hot_columns = [
        column
        for column in ["ProductCD", "card4", "card6", "M6"]
        if column in data.columns
    ]
    data = pd.get_dummies(
        data,
        columns=one_hot_columns,
        drop_first=True,
        dtype=int,
    )

    return data


def build_pipeline() -> ImbPipeline:
    """Create the scaler → SMOTE → XGBoost training pipeline."""
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", model),
        ]
    )


def main() -> None:
    """Run the complete baseline training workflow."""
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print(classification_report(
        y_test,
        predictions,
        target_names=["Not Fraud", "Fraud"],
        digits=2,
    ))


if __name__ == "__main__":
    main()
