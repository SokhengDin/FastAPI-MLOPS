import pickle
import pandas as pd
import numpy as np
import mlflow

from pipeline.preprocess_pipeline import PreprocessResult


def load_artifacts() -> tuple:
    with open("models/preprocessor.pkl", "rb") as f:
        pre: PreprocessResult = pickle.load(f)

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    return model, pre


@mlflow.trace(name="preprocess_input", span_type="PARSER")
def _preprocess_input(features: dict, pre: PreprocessResult) -> pd.DataFrame:
    X = pd.DataFrame([features])
    for col in pre.cate_cols:
        if col in X.columns:
            X[col] = pre.col_encoders[col].transform(X[col].astype(str))
    return X


@mlflow.trace(name="predict", span_type="LLM")
def predict(
    features    : dict
    , model
    , pre       : PreprocessResult
) -> dict:
    X           = _preprocess_input(features, pre)
    pred        = model.predict(X)[0]
    label       = pre.le_target.inverse_transform([pred])[0]
    proba       = model.predict_proba(X)[0]
    classes     = pre.le_target.classes_

    return {
        "irrigation_need"   : label
        , "confidence"      : float(proba[pred])
        , "probabilities"   : [
            {"label": cls, "probability": float(p)}
            for cls, p in zip(classes, proba)
        ]
    }


def predict_batch(
    test_df     : pd.DataFrame
    , model
    , pre       : PreprocessResult
) -> tuple[np.ndarray, pd.Series]:
    ids     = test_df["id"]
    X_test  = test_df.drop(columns=["id"], errors="ignore")

    for col in pre.cate_cols:
        X_test[col] = pre.col_encoders[col].transform(X_test[col].astype(str))

    preds   = model.predict(X_test)
    labels  = pre.le_target.inverse_transform(preds)

    return labels, ids
