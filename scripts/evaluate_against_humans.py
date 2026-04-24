from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


MODEL_CSV = Path("outputs/main_eval/main_eval_results.csv")
HUMAN_CSV = Path("outputs/main_eval/human_scores.csv")
OUT_DIR = Path("outputs/main_eval")
MERGED_CSV = OUT_DIR / "final_merged.csv"
SUMMARY_CSV = OUT_DIR / "human_evaluation_summary.csv"
OVERALL_CSV = OUT_DIR / "human_evaluation_overall.csv"
REPORT_MD = OUT_DIR / "human_evaluation_report.md"

DIMENSIONS = ("confidence", "clarity", "engagement")
VARIANTS = ("speech_only", "text_only", "multimodal")


def _safe_corr(method, x: pd.Series, y: pd.Series) -> float | None:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return None
    value = method(x, y)
    return float(value.statistic if hasattr(value, "statistic") else value.correlation)


def _metrics(sub: pd.DataFrame, dimension: str) -> dict:
    model_col = f"{dimension}_scaled"
    human_col = f"human_{dimension}"
    error = sub[model_col] - sub[human_col]
    abs_error = error.abs()

    return {
        "n": int(len(sub)),
        "spearman": _safe_corr(spearmanr, sub[model_col], sub[human_col]),
        "pearson": _safe_corr(pearsonr, sub[model_col], sub[human_col]),
        "mae": float(abs_error.mean()),
        "rmse": float((error.pow(2).mean()) ** 0.5),
        "bias_model_minus_human": float(error.mean()),
        "within_0_5": float((abs_error <= 0.5).mean()),
        "within_1_0": float((abs_error <= 1.0).mean()),
        "model_mean": float(sub[model_col].mean()),
        "human_mean": float(sub[human_col].mean()),
    }


def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def build_evaluation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = pd.read_csv(MODEL_CSV)
    human_df = pd.read_csv(HUMAN_CSV)

    model_df["recording_id"] = model_df["filename"].str.replace(".wav", "", regex=False)
    human_df = human_df.rename(
        columns={
            "confidence_human": "human_confidence",
            "clarity_human": "human_clarity",
            "engagement_human": "human_engagement",
        }
    )

    merged = model_df.merge(human_df, on="recording_id", how="inner")
    expected_rows = len(human_df) * len(VARIANTS)
    if len(merged) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} merged rows, got {len(merged)}.")

    for dimension in DIMENSIONS:
        merged[f"{dimension}_scaled"] = 1.0 + (merged[dimension] / 100.0) * 4.0
        merged[f"{dimension}_error"] = merged[f"{dimension}_scaled"] - merged[f"human_{dimension}"]
        merged[f"{dimension}_abs_error"] = merged[f"{dimension}_error"].abs()

    rows = []
    for variant in VARIANTS:
        sub = merged[merged["variant"] == variant]
        for dimension in DIMENSIONS:
            rows.append(
                {
                    "variant": variant,
                    "dimension": dimension,
                    **_metrics(sub, dimension),
                }
            )

    summary = pd.DataFrame(rows)
    overall = (
        summary.groupby("variant", as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            mean_pearson=("pearson", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias_model_minus_human=("bias_model_minus_human", "mean"),
            mean_within_0_5=("within_0_5", "mean"),
            mean_within_1_0=("within_1_0", "mean"),
        )
        .sort_values(["mean_mae", "mean_spearman"], ascending=[True, False])
    )

    return merged, summary, overall


def write_report(summary: pd.DataFrame, overall: pd.DataFrame) -> None:
    best_mae = overall.iloc[0]
    best_corr = overall.sort_values("mean_spearman", ascending=False).iloc[0]

    lines = [
        "# Main Evaluation Against Human Scores",
        "",
        "Model scores are linearly scaled from 0-100 onto the same 1-5 range as the human scores.",
        "Lower MAE/RMSE is better. Higher Spearman/Pearson correlation is better.",
        "",
        "## Overall By Variant",
        "",
        "| Variant | Mean Spearman | Mean Pearson | Mean MAE | Mean RMSE | Mean Bias | Within 0.5 | Within 1.0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in overall.iterrows():
        lines.append(
            "| {variant} | {spearman} | {pearson} | {mae} | {rmse} | {bias} | {within_05} | {within_10} |".format(
                variant=row["variant"],
                spearman=_format_float(row["mean_spearman"]),
                pearson=_format_float(row["mean_pearson"]),
                mae=_format_float(row["mean_mae"]),
                rmse=_format_float(row["mean_rmse"]),
                bias=_format_float(row["mean_bias_model_minus_human"]),
                within_05=_format_float(row["mean_within_0_5"]),
                within_10=_format_float(row["mean_within_1_0"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Lowest average error: `{best_mae['variant']}` with mean MAE {_format_float(best_mae['mean_mae'])}.",
            f"- Highest average rank correlation: `{best_corr['variant']}` with mean Spearman {_format_float(best_corr['mean_spearman'])}.",
            "- Positive bias means the model scores higher than the human scores on average; negative bias means it scores lower.",
            "",
            "## Per Dimension",
            "",
            "| Variant | Dimension | Spearman | Pearson | MAE | RMSE | Bias | Within 0.5 | Within 1.0 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in summary.iterrows():
        lines.append(
            "| {variant} | {dimension} | {spearman} | {pearson} | {mae} | {rmse} | {bias} | {within_05} | {within_10} |".format(
                variant=row["variant"],
                dimension=row["dimension"],
                spearman=_format_float(row["spearman"]),
                pearson=_format_float(row["pearson"]),
                mae=_format_float(row["mae"]),
                rmse=_format_float(row["rmse"]),
                bias=_format_float(row["bias_model_minus_human"]),
                within_05=_format_float(row["within_0_5"]),
                within_10=_format_float(row["within_1_0"]),
            )
        )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    merged, summary, overall = build_evaluation()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    overall.to_csv(OVERALL_CSV, index=False)
    write_report(summary, overall)

    print(f"Saved merged file: {MERGED_CSV}")
    print(f"Saved per-dimension summary: {SUMMARY_CSV}")
    print(f"Saved overall summary: {OVERALL_CSV}")
    print(f"Saved report: {REPORT_MD}")
    print()
    print(overall.to_string(index=False))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
