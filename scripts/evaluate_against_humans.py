import pandas as pd
from scipy.stats import spearmanr

MODEL_CSV = "outputs/main_eval/main_eval_results.csv"
HUMAN_CSV = "outputs/main_eval/human_scores.csv"
OUT_CSV = "outputs/main_eval/final_merged.csv"


def corr(x, y):
    return spearmanr(x, y).correlation


def main():
    model_df = pd.read_csv(MODEL_CSV)
    human_df = pd.read_csv(HUMAN_CSV)

    print("Model columns:", list(model_df.columns))
    print("Human columns:", list(human_df.columns))
    print()

    # rename human columns to a consistent format
    human_df = human_df.rename(columns={
        "confidence_human": "human_confidence",
        "clarity_human": "human_clarity",
        "engagement_human": "human_engagement",
    })

    merged = model_df.merge(human_df, on="recording_id")

    # scale model scores to 1–5
    for col in ["confidence", "clarity", "engagement"]:
        merged[f"{col}_scaled"] = (merged[col] / 100.0) * 5.0

    merged.to_csv(OUT_CSV, index=False)
    print("Saved merged file:", OUT_CSV)
    print()

    for variant in ["multimodal", "speech_only", "text_only"]:
        sub = merged[merged["variant"] == variant]

        print(f"--- {variant} ---")
        print("confidence:", corr(sub["confidence_scaled"], sub["human_confidence"]))
        print("clarity:", corr(sub["clarity_scaled"], sub["human_clarity"]))
        print("engagement:", corr(sub["engagement_scaled"], sub["human_engagement"]))
        print()


if __name__ == "__main__":
    main()