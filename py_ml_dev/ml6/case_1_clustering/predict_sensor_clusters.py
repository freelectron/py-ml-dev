"""
Runs semi-supervised label propagation on sensor data and writes predictions to CSV.

Usage:
    python predict_sensor_clusters.py [--config config.yaml]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------------

def check_data(df: pd.DataFrame, label_col: str, focus_sensors: list[str]) -> None:
    missing_sensors = [s for s in focus_sensors if s not in df.columns]
    assert not missing_sensors, f"Focus sensors missing from data: {missing_sensors}"

    assert label_col in df.columns, f"Label column '{label_col}' not found in data"

    assert len(df) > 0, "Input data is empty"

    # At least some labeled rows are required to train
    n_labeled = df[label_col].notna().sum()
    assert n_labeled > 0, "No labeled rows found — cannot train"

    # At least two distinct classes are needed for classification
    n_classes = df[label_col].dropna().nunique()
    assert n_classes >= 2, f"Need at least 2 labeled classes, found {n_classes}"

    # Focus sensors must be numeric and have no nulls (model cannot handle them)
    for s in focus_sensors:
        assert pd.api.types.is_numeric_dtype(df[s]), f"Sensor column '{s}' is not numeric"
        n_null = df[s].isna().sum()
        assert n_null == 0, f"Sensor column '{s}' has {n_null} null values"

    # Labels on labeled rows must be numeric (SVC encodes them as floats in this pipeline)
    labeled_labels = df[label_col].dropna()
    assert pd.api.types.is_numeric_dtype(labeled_labels), \
        f"Label column '{label_col}' must be numeric"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_radius_feature(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].values
    R = np.sqrt(np.sum(X * X, axis=1)).reshape(-1, 1)
    return np.concatenate([X, R], axis=1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fit_predict(
    df: pd.DataFrame,
    label_col: str,
    focus_sensors: list[str],
    svc_gamma: float,
    svc_random_state: int,
) -> pd.DataFrame:
    # Use Sensor 9 (index 1) and radius R (index 3) as features — established during exploration
    X = add_radius_feature(df, focus_sensors)[:, [1, 3]]

    df = df.copy()
    df["_label_ssl"] = df[label_col].fillna(-1)
    y = df["_label_ssl"].values

    clf = SelfTrainingClassifier(
        SVC(probability=True, random_state=svc_random_state, gamma=svc_gamma)
    ).fit(X, y)

    y_proba = clf.predict_proba(X)
    proba_cols = {f"Probability_label_{c}": y_proba[:, i] for i, c in enumerate(clf.classes_)}
    predicted = clf.classes_[y_proba.argmax(axis=1)]

    df = df.drop(columns=["_label_ssl"])
    for col, vals in proba_cols.items():
        df[col] = vals
    df["Predicted_Label"] = predicted

    return df


# ---------------------------------------------------------------------------
# Sanity check on output
# ---------------------------------------------------------------------------

def check_predictions(df: pd.DataFrame, label_col: str) -> None:
    mismatched = df[(df[label_col] != df["Predicted_Label"]) & df[label_col].notna()]
    assert len(mismatched) == 0, (
        f"{len(mismatched)} labeled rows have a predicted label that differs from the ground truth"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sensor clusters via semi-supervised SVC")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    input_path = Path(data_cfg["input_path"])
    output_path = Path(data_cfg["output_path"])
    label_col = data_cfg["label_column"]
    focus_sensors: list[str] = data_cfg["focus_sensors"]

    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    print("Validating data assumptions...")
    check_data(df, label_col, focus_sensors)

    print("Fitting model and predicting...")
    df_out = fit_predict(
        df,
        label_col=label_col,
        focus_sensors=focus_sensors,
        svc_gamma=model_cfg["svc_gamma"],
        svc_random_state=model_cfg["svc_random_state"],
    )

    print("Checking predictions against ground truth...")
    check_predictions(df_out, label_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")
    print(f"  Rows: {len(df_out)}")
    print(f"  Class distribution:\n{df_out['Predicted_Label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
