from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    df: pd.DataFrame,
    ycol: str = "gpa",
    corr_thr: float = 0.98,              # feature-feature pruning threshold
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    # 0) Basic checks + NA handling
    if ycol not in df.columns:
        raise ValueError(f"Target column '{ycol}' not found in dataframe")
    df = df.dropna()
    print(f"[dropna] shape: {df.shape}")

    # 1) Split target and numeric features
    y = df[ycol]
    X = df.select_dtypes(include=["number"]).drop(columns=[ycol])
    print(f"[numeric selection] X shape: {X.shape}")
    print("[numeric selection] Initial numeric columns:", list(X.columns))

    # 2) Drop constant columns
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print("[constants] Dropping constant columns:", const_cols)
        X = X.drop(columns=const_cols)
    print(f"[constants] X shape after drop: {X.shape}")

    # 3) Drop exact duplicate columns (keep first occurrence)
    dup_cols: List[str] = []
    seen = {}
    for col in X.columns:
        sig = tuple(X[col].values.tolist())
        if sig in seen:
            dup_cols.append(col)
        else:
            seen[sig] = col
    if dup_cols:
        print("[duplicates] Dropping exact duplicate columns:", dup_cols)
        X = X.drop(columns=dup_cols)
    print(f"[duplicates] X shape after drop: {X.shape}")

    # 4) Prune highly correlated feature pairs (keep the one more correlated with y)
    if X.shape[1] >= 2:
        corr_abs = X.corr().abs()
        ycorr_abs = X.corrwith(y).abs()
        to_drop = set()

        cols = list(X.columns)
        for i in range(len(cols)):
            ci = cols[i]
            if ci in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                cj = cols[j]
                if cj in to_drop:
                    continue
                r = corr_abs.iat[i, j]
                if np.isfinite(r) and r >= corr_thr:
                    ri = float(ycorr_abs.get(ci, 0.0))
                    rj = float(ycorr_abs.get(cj, 0.0))
                    drop = cj if ri >= rj else ci
                    to_drop.add(drop)
                    print(f"[high corr] {ci} ~ {cj} |r|={r:.3f} -> drop {drop} "
                          f"(|corr with {ycol}|: {ci}={ri:.3f}, {cj}={rj:.3f})")

        if to_drop:
            X = X.drop(columns=list(to_drop))
        print(f"[high corr] X shape after pruning: {X.shape}")
    else:
        print("[high corr] Skipped (fewer than 2 columns).")

    # 5) Print remaining columns and their stats
    remaining_cols = list(X.columns)
    print("[remain] Columns:", remaining_cols)
    print("[remain] Describe():")
    print(X.describe())

    # 6) Drop features overly correlated with target (hard-coded at 0.6)
    ycorr_final = X.corrwith(y)
    to_drop_y = ycorr_final[ycorr_final.abs() >= 0.5].index.tolist()
    if to_drop_y:
        print(f"[target corr] Dropping features with |corr({ycol})| ≥ 0.6: {to_drop_y}")
        X = X.drop(columns=to_drop_y)
    print(f"[target corr] X shape after drop: {X.shape}")


    # 7) Scale X; center y
    x_scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = pd.DataFrame(
        x_scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    y_centered = y - y.mean()
    feature_names = list(X_scaled.columns)
    print(f"[final] X_scaled shape: {X_scaled.shape}; y centered std: {float(y_centered.std()):.6f}")

    return X_scaled, y_centered, feature_names

def load_lsac() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    local_path = os.environ.get("LSAC_CSV_PATH", None)
    if local_path and os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        url = "https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv"
        df = pd.read_csv(url)
    return preprocess_data(df)
