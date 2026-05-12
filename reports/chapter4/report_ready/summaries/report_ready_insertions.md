# Report-Ready Chapter 4 Insertions

## Main Report Tables

- Table 4.1: Evaluation and inter-rater summary
- Table 4.2: Overall performance compact
- Table 4.3: Ablation summary compact
- Table 4.4: Top feature correlations compact
- Table 4.5: Case studies compact
- Table 4.6: Penalty effect compact
- Table 4.7: Error cases compact
- Table 4.8: Cold-start latency summary

## Main Report Figures

- Figure 4.1: MAE by variant and dimension
- Figure 4.2: Correlations by variant and dimension
- Figure 4.3: Multimodal scatter combined
- Figure 4.4: Top feature correlations
- Figure 4.5: Cold-start latency by stage
- Figure 4.6: Multimodal error distribution
- Figure 4.x: Inter-rater variability dot plot

## Appendix / Notebook Only

- `appendix_gender_descriptive_summary.csv/md`
- `raw_cold_start_latency_logs.csv/md`
- `gender_descriptive_notes.md`
- `emotion_analysis_status.md`
- Full human baseline, variant-definition table, p-values, dimension summary, full feature correlation table, full case-study feedback excerpts, and original cached latency logs remain notebook/audit or appendix material only.

## Warnings

- The evaluation uses 20 recordings from 10 speakers, so quantitative claims should be framed cautiously.
- The gender subgroup summary is descriptive only because each group contains five speakers; no significance testing or fairness conclusions are included.
- The old latency logs used cached transcription. Table 4.8 uses a new cache-disabled benchmark and documents one-time model loading separately where measurable.
- Cold-start transcription-result caching was disabled; the loaded transcriber model was reused in-process if model loading succeeded.
- Emotion analysis remains supplementary only and is not added to the primary MAE/Pearson/Spearman comparison.
