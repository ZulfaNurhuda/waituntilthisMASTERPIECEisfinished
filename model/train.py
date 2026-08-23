import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from features import extract_features, FEATURE_COLUMNS

DATA_PATH = "synthetic_dataset.csv"
MODEL_PATH = "shelf_life_model.joblib"
QUANTILE_LOW_PATH = "shelf_life_model_q10.joblib"
QUANTILE_HIGH_PATH = "shelf_life_model_q90.joblib"


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        temps = json.loads(row["temp_history_celsius"])
        feats = extract_features(temps)
        rows.append(feats)
    feat_df = pd.DataFrame(rows)
    return feat_df[FEATURE_COLUMNS]


def main():
    df = pd.read_csv(DATA_PATH)
    X = build_feature_table(df)
    y = df["true_remaining_shelf_life_hours"].values

    X_train, X_test, y_train, y_test, scen_train, scen_test = train_test_split(
        X, y, df["scenario"], test_size=0.2, random_state=42
    )

    # Point-estimate model (median-ish behavior via default squared error loss)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)

    model_q10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.10, n_estimators=300, max_depth=3,
        learning_rate=0.05, random_state=42
    )
    model_q90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=300, max_depth=3,
        learning_rate=0.05, random_state=42
    )
    model_q10.fit(X_train, y_train)
    model_q90.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: {mae:.1f} hours ({mae/24:.2f} days)")
    print(f"Test R^2: {r2:.4f}")

    results = pd.DataFrame({
        "scenario": scen_test.values, "y_true": y_test, "y_pred": preds
    })
    print("\nPer-scenario MAE (hours):")
    print(results.groupby("scenario").apply(
        lambda g: mean_absolute_error(g["y_true"], g["y_pred"])
    ))

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(model_q10, QUANTILE_LOW_PATH)
    joblib.dump(model_q90, QUANTILE_HIGH_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved quantile models -> {QUANTILE_LOW_PATH}, {QUANTILE_HIGH_PATH}")


if __name__ == "__main__":
    main()
