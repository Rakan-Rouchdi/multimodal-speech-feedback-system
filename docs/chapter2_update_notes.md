# Chapter 2 Update Notes

| Chapter 2 section likely affected | What changed | What text should be updated | Priority |
|---|---|---|---|
| Methodology overview | Pipeline now explicitly supports `speech_only`, `text_only`, and `multimodal` variants. | Describe the three variants and why they support ablation/comparison. | High |
| Transcription approach | The implemented system uses `nyrahealth/faster_CrisperWhisper` via `faster-whisper`, not standard Whisper. | Replace any generic Whisper wording with CrisperWhisper/faster-whisper wording. | High |
| Text feature definitions | Added ASR-safe `avg_clause_length`, `estimated_clause_count`, and `lexical_diversity`; readability no longer depends only on full stops. | Update linguistic feature table and explain punctuation-independent clause chunking. | High |
| Acoustic feature definitions | Pause ratio and pause rate are derived in the pipeline and used in scoring/evaluation. | Include `pause_ratio = total_pause_sec / duration_sec` and `pause_rate_per_min`. | Medium |
| Scoring strategy | Thresholds/weights were refined and multimodal consistency penalty added. | Replace old formulas/weights with final values from `docs/scoring_formulas_and_thresholds.md`. | High |
| Methodology justification | Human-score comparison was used to refine formula thresholds. | State that formulas are interpretable heuristics refined against available human ratings; avoid claiming objective correctness. | High |
| Evaluation design | Human evaluation outputs now include Spearman, Pearson, MAE, RMSE, bias, within-0.5, within-1.0. | Update evaluation metrics description. | Medium |
| Caching/performance assumptions | Transcription cache is implemented and enabled by default in batch runs. | Mention caching as a performance optimisation, not as a model component. | Medium |
| Emotion model | Emotion model is optional and not enabled in current main evaluation outputs. | Do not imply emotion was used in all results unless rerun with `--use_emotion`. | High |
| Figures/diagrams | Architecture and output figures now exist in `docs/figures/`. | Align Chapter 2 methodology diagram with the final implemented modules if reused. | Medium |
