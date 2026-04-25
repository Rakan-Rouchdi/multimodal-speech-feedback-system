# Consistency Audit

| Inconsistency found | Risk to dissertation accuracy | Recommended fix | Status |
|---|---|---|---|
| Some old output aliases were removed from code; docs needed to reflect canonical fields only. | Could describe stale JSON fields such as `latency_timings` or top-level score aliases. | Use `latency_ms`, nested `scores`, `speech_metrics`, `text_metrics`. | Fixed in documentation. |
| Current outputs have `emotion_output: null`. | Could falsely imply emotion model contributed to reported main evaluation scores. | State emotion is optional and disabled unless `--use_emotion` is used. | Fixed in documentation. |
| CrisperWhisper punctuation is sparse. | Sentence/readability claims could be inaccurate. | Use ASR-safe chunking and describe it as heuristic. | Fixed in code/docs/tests. |
| Emotion dataset name is not explicitly cited in notebook. | Dataset citation could be fabricated. | Add formal dataset citation manually if known. | TODO: Needs confirmation. |
| On-disk `best_model.h5` provenance is not cryptographically tied to notebook checkpoint. | Overclaiming exact provenance. | Say runtime model path exists; exact provenance needs confirmation. | TODO: Needs confirmation. |
| Scoring formulas were refined using `main_eval` human scores. | Could overstate generalisation. | Describe as in-sample human-score refinement; require held-out set for generalisation. | Fixed in documentation. |
| `schema/output_schema.json` is representative, not programmatically enforced. | Could claim formal validation when none exists. | Say representative output contract, not enforced JSON Schema. | Fixed in documentation. |
| `pause_rate_score` is returned as a subscore but not used in headline formulas. | Could list it incorrectly as a weighted formula input. | Document as calculated but unused in headline formulas. | Fixed in scoring documentation. |
| Real CrisperWhisper test is skipped by default. | Could overstate automated E2E coverage. | Mark as optional integration test requiring env var. | Fixed in testing docs. |
