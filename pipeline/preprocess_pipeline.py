import yaml
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from api import logger


@dataclass
class PreprocessResult:
    X_train         : pd.DataFrame
    X_val           : pd.DataFrame
    y_train         : np.ndarray
    y_val           : np.ndarray
    sample_weights  : np.ndarray
    le_target       : LabelEncoder
    col_encoders    : dict  = field(default_factory=dict)
    cate_cols       : list  = field(default_factory=list)


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def run(params: dict) -> PreprocessResult:
    # Prepare configurate for data pipeline
    data_cfg        = params["data"]
    preprocess_cfg  = params["preprocess"]

    df          = pd.read_csv(data_cfg["train_path"])
    target      = data_cfg["target"]
    drop_cols   = data_cfg["drop_cols"]

    y_raw   = df[target]
    X       = df.drop(columns=drop_cols, errors="ignore")

    le_target   = LabelEncoder()
    y           = le_target.fit_transform(y_raw)

    cate_cols       = X.select_dtypes(include="object").columns.tolist()
    col_encoders    = {}

    for col in cate_cols:
        enc             = LabelEncoder()
        X[col]          = enc.fit_transform(X[col].astype(str))
        col_encoders[col] = enc

    X_train, X_val, y_train, y_val = train_test_split(
        X, y
        , test_size     = preprocess_cfg["test_size"]
        , random_state  = preprocess_cfg["seed"]
        , stratify      = y
    )

    sample_weights = compute_sample_weight("balanced", y_train)

    return PreprocessResult(
        X_train         = X_train
        , X_val         = X_val
        , y_train       = y_train
        , y_val         = y_val
        , sample_weights = sample_weights
        , le_target     = le_target
        , col_encoders  = col_encoders
        , cate_cols     = cate_cols
    )


if __name__ == "__main__":
    params  = load_params()
    result  = run(params)

    Path(params["artifacts"]["model_dir"]).mkdir(exist_ok=True)

    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(result, f)

    logger.info(f"X_train: {result.X_train.shape} | X_val: {result.X_val.shape}")
    logger.info(f"Classes: {result.le_target.classes_}")
