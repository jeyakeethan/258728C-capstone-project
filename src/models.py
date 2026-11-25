from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


@dataclass
class ModelResult:
    model_name: str
    mae: float
    rmse: float
    r2: float


def chronological_split(df, train_ratio=0.70, validation_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def make_model(name, hgb_params=None):
    name = name.lower()
    if name == "linear":
        estimator = LinearRegression()
    elif name in {"hgb", "hist_gradient_boosting"}:
        params = {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
            "random_state": 42,
        }
        if hgb_params:
            params.update(hgb_params)
        estimator = HistGradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {name}")

    # Missing weather is a property of the raw UNISOLAR sample, not a reason
    # to drop the corresponding target observations. Median imputation is fit
    # only on the training partition via this Pipeline, preventing leakage.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", estimator),
    ])


def metrics_from_predictions(y_true, pred, model_name):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return ModelResult(
        model_name=model_name,
        mae=float(mean_absolute_error(y_true, pred)),
        rmse=float(mean_squared_error(y_true, pred) ** 0.5),
        r2=float(r2_score(y_true, pred)),
    )


def evaluate(model, X, y, model_name):
    pred = model.predict(X)
    return metrics_from_predictions(y, pred, model_name)


def evaluate_persistence(frame, lag_column="solar_power_lag_24", model_name="persistence_24h"):
    """Seasonal persistence: predict this timestamp using solar from 24 h earlier."""
    if lag_column not in frame.columns:
        raise ValueError(f"Persistence column {lag_column!r} is missing")
    valid = frame[["solar_power", lag_column]].dropna()
    return metrics_from_predictions(valid["solar_power"], valid[lag_column], model_name)


def save_bundle(model, feature_columns, path, metadata=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": feature_columns,
        "metadata": metadata or {},
    }, path)


def load_bundle(path):
    return joblib.load(path)
