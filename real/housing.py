from __future__ import annotations
import numpy as np, pandas as pd
from typing import Tuple, List, Optional
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    center_y: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X.values),
        columns=X.columns,
        index=X.index
    )
    if y is None or not center_y:
        return X_scaled, y
    y_centered = y - float(y.mean())
    return X_scaled, y_centered

def load_housing(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    data = fetch_california_housing(as_frame=as_frame)
    if as_frame:
        X = data.frame.drop(columns=[data.target.name])
        y = data.frame[data.target.name]
        feature_names = list(X.columns)
    else:
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="MedHouseVal")
        feature_names = data.feature_names
    Xp, yp = preprocess_data(X, y, center_y=True)
    return Xp, yp, feature_names