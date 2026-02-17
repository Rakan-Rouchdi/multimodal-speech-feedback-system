from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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

    # Means
    mean_df = df.groupby("variant")[["confidence", "clarity", "engagement", "lat_total_ms"]].mean().round(2)
    mean_out = out_dir / "mean_scores_by_variant.csv"
    mean_df.to_csv(mean_out)

    # Std dev (useful for consistency)
    std_df = df.groupby("variant")[["confidence", "clarity", "engagement", "lat_total_ms"]].std().round(2)
    std_out = out_dir / "std_scores_by_variant.csv"
    std_df.to_csv(std_out)

    print("Saved:", mean_out)
    print(mean_df)
    print("\nSaved:", std_out)
    print(std_df)

    # Plots (one chart per metric)
    for col, title, ylabel, fname in [
        ("confidence", "Mean Confidence by Variant", "Confidence Score", "mean_confidence.png"),
        ("clarity", "Mean Clarity by Variant", "Clarity Score", "mean_clarity.png"),
        ("engagement", "Mean Engagement by Variant", "Engagement Score", "mean_engagement.png"),
        ("lat_total_ms", "Mean Total Latency by Variant", "Latency (ms)", "mean_latency.png"),
    ]:
        plt.figure()
        mean_df[col].plot(kind="bar")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Variant")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    print("\nSaved plots to:", out_dir)


if __name__ == "__main__":
    main()
