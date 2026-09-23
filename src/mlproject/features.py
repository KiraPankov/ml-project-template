import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.config import DataConfig


def infer_feature_columns(df: pd.DataFrame, cfg: DataConfig) -> tuple[list[str], list[str]]:
    excluded = {cfg.target, cfg.time_column}
    numeric = cfg.numeric_features or [
        c for c in df.select_dtypes("number").columns if c not in excluded
    ]
    categorical = cfg.categorical_features or [
        c for c in df.select_dtypes(exclude="number").columns if c not in excluded
    ]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)]
    )
