from __future__ import annotations

import os
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


DATA_PATH = "outputs/main_eval/final_merged.csv"
OUT_DIR = "outputs/main_eval/ml"

FEATURES = [
    "duration_sec",
    "energy_mean",
    "pause_count",
    "mean_pause_sec",
    "total_pause_sec",
    "pitch_mean_hz",
    "pitch_std_hz",
    "speech_rate_wpm",
    "word_count",
    "filler_count",
    "filler_rate_per_100w",
    "repeat_rate",
    "readability_proxy",
]

TARGETS = [
    "human_confidence",
    "human_clarity",
    "human_engagement",
]


def safe_spearman(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred)
    return corr


def evaluate_target(df: pd.DataFrame, target: str):
    """
    Leave-one-speaker-out evaluation.
    Stores out-of-fold predictions for:
    - heuristic baseline
    - linear regression
    - random forest
    """
    speakers = sorted(df["speaker_id"].unique())
    all_preds = []

    for held_out_speaker in speakers:
        train_df = df[df["speaker_id"] != held_out_speaker].copy()
        test_df = df[df["speaker_id"] == held_out_speaker].copy()

        X_train = train_df[FEATURES].fillna(0)
        y_train = train_df[target]
        X_test = test_df[FEATURES].fillna(0)
        y_test = test_df[target]

        heuristic_col = target.replace("human_", "") + "_scaled"
        heuristic_pred = test_df[heuristic_col].values

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=5
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        fold_df = pd.DataFrame({
            "recording_id": test_df["recording_id"].values,
            "speaker_id": test_df["speaker_id"].values,
            "target": target,
            "y_true": y_test.values,
            "heuristic_pred": heuristic_pred,
            "lr_pred": lr_pred,
            "rf_pred": rf_pred,
        })

        all_preds.append(fold_df)

    return pd.concat(all_preds, ignore_index=True)


def summarize_predictions(pred_df: pd.DataFrame):
    """
    Compute one overall Spearman + MAE for each target,
    using all out-of-fold predictions combined.
    """
    rows = []

    for target in TARGETS:
        sub = pred_df[pred_df["target"] == target].copy()

        rows.append({
            "target": target,

            "heuristic_spearman": safe_spearman(sub["y_true"], sub["heuristic_pred"]),
            "heuristic_mae": mean_absolute_error(sub["y_true"], sub["heuristic_pred"]),

            "lr_spearman": safe_spearman(sub["y_true"], sub["lr_pred"]),
            "lr_mae": mean_absolute_error(sub["y_true"], sub["lr_pred"]),

            "rf_spearman": safe_spearman(sub["y_true"], sub["rf_pred"]),
            "rf_mae": mean_absolute_error(sub["y_true"], sub["rf_pred"]),
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Start with multimodal rows only
    df = df[df["variant"] == "multimodal"].copy()

    # Scale heuristic scores from 0–100 to 1–5
    for col in ["confidence", "clarity", "engagement"]:
        df[f"{col}_scaled"] = (df[col] / 100.0) * 5.0

    all_pred_frames = []

    for target in TARGETS:
        pred_df = evaluate_target(df, target)
        all_pred_frames.append(pred_df)

    predictions = pd.concat(all_pred_frames, ignore_index=True)
    summary = summarize_predictions(predictions)

    predictions.to_csv(f"{OUT_DIR}/ml_out_of_fold_predictions.csv", index=False)
    summary.to_csv(f"{OUT_DIR}/ml_comparison_summary.csv", index=False)

    print("Saved:")
    print(f" - {OUT_DIR}/ml_out_of_fold_predictions.csv")
    print(f" - {OUT_DIR}/ml_comparison_summary.csv")
    print()
    print("=== Overall results across all out-of-fold predictions ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()