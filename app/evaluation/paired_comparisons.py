from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


SCORES = ["confidence", "clarity", "engagement"]


def pivot_scores(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """
    Returns a table with one row per filename and one column per variant for a given score.
    """
    pivot = df.pivot_table(index="filename", columns="variant", values=score, aggfunc="mean")
    # Ensure expected columns exist
    for v in ["speech_only", "text_only", "multimodal"]:
        if v not in pivot.columns:
            pivot[v] = pd.NA
    return pivot[["speech_only", "text_only", "multimodal"]]


def paired_tests(a: pd.Series, b: pd.Series):
    """
    Paired t-test and Wilcoxon (when possible). Drops NA pairs.
    Returns dict with test stats.
    """
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3:
        return {
            "n": len(pair),
            "t_stat": None,
            "t_p": None,
            "wilcoxon_stat": None,
            "wilcoxon_p": None,
        }

    x = pair.iloc[:, 0].astype(float)
    y = pair.iloc[:, 1].astype(float)

    t_stat, t_p = stats.ttest_rel(x, y)

    # Wilcoxon requires not all differences = 0
    diffs = (x - y)
    if (diffs == 0).all():
        w_stat, w_p = None, None
    else:
        try:
            w_stat, w_p = stats.wilcoxon(x, y)
        except ValueError:
            w_stat, w_p = None, None

    return {
        "n": len(pair),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcoxon_stat": None if w_stat is None else float(w_stat),
        "wilcoxon_p": None if w_p is None else float(w_p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="outputs/eval_summary_own.csv")
    parser.add_argument("--out_dir", type=str, default="outputs/analysis")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    rows = []
    delta_tables = []

    for score in SCORES:
        piv = pivot_scores(df, score)

        # Deltas
        piv[f"delta_multimodal_minus_speech_only_{score}"] = piv["multimodal"] - piv["speech_only"]
        piv[f"delta_multimodal_minus_text_only_{score}"] = piv["multimodal"] - piv["text_only"]

        delta_tables.append(piv[[f"delta_multimodal_minus_speech_only_{score}",
                                 f"delta_multimodal_minus_text_only_{score}"]])

        # Mean deltas + percent improvements (relative to baseline mean)
        mean_multi = piv["multimodal"].mean()
        mean_speech = piv["speech_only"].mean()
        mean_text = piv["text_only"].mean()

        mean_delta_vs_speech = (piv["multimodal"] - piv["speech_only"]).mean()
        mean_delta_vs_text = (piv["multimodal"] - piv["text_only"]).mean()

        pct_vs_speech = None if mean_speech == 0 else (mean_delta_vs_speech / mean_speech) * 100.0
        pct_vs_text = None if mean_text == 0 else (mean_delta_vs_text / mean_text) * 100.0

        # Paired tests (multimodal vs speech_only, multimodal vs text_only)
        tests_vs_speech = paired_tests(piv["multimodal"], piv["speech_only"])
        tests_vs_text = paired_tests(piv["multimodal"], piv["text_only"])

        rows.append({
            "score": score,
            "mean_multimodal": round(float(mean_multi), 2),
            "mean_speech_only": round(float(mean_speech), 2),
            "mean_text_only": round(float(mean_text), 2),
            "mean_delta_multi_minus_speech": round(float(mean_delta_vs_speech), 2),
            "mean_delta_multi_minus_text": round(float(mean_delta_vs_text), 2),
            "pct_improvement_vs_speech": None if pct_vs_speech is None else round(float(pct_vs_speech), 2),
            "pct_improvement_vs_text": None if pct_vs_text is None else round(float(pct_vs_text), 2),

            "n_vs_speech": tests_vs_speech["n"],
            "t_p_vs_speech": tests_vs_speech["t_p"],
            "wilcoxon_p_vs_speech": tests_vs_speech["wilcoxon_p"],

            "n_vs_text": tests_vs_text["n"],
            "t_p_vs_text": tests_vs_text["t_p"],
            "wilcoxon_p_vs_text": tests_vs_text["wilcoxon_p"],
        })

        # Save per-score pivot table too
        piv_out = out_dir / f"pivot_{score}.csv"
        piv.to_csv(piv_out)

    summary = pd.DataFrame(rows)
    summary_out = out_dir / "paired_comparison_summary.csv"
    summary.to_csv(summary_out, index=False)

    # Also save a combined delta table (one row per file)
    deltas = pd.concat(delta_tables, axis=1)
    deltas_out = out_dir / "per_file_deltas.csv"
    deltas.to_csv(deltas_out)

    print("Saved:", summary_out)
    print(summary)
    print("\nSaved:", deltas_out)
    print("\nSaved pivot tables: pivot_confidence.csv / pivot_clarity.csv / pivot_engagement.csv")


if __name__ == "__main__":
    main()
