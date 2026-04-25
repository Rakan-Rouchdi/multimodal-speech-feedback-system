# Multimodal Speech Feedback System

Final Year BSc Computer Science with AI dissertation project.

This repository implements a local multimodal speech feedback pipeline. It analyses an audio file and generates confidence, clarity, and engagement feedback using acoustic features, CrisperWhisper transcription, linguistic features, optional emotion prediction, heuristic scoring, and structured JSON logging.

## What The System Does

The implemented pipeline supports:

- audio preprocessing
- acoustic feature extraction
- CrisperWhisper transcription using `nyrahealth/faster_CrisperWhisper`
- transcript cleaning and disfluency preservation
- linguistic feature extraction
- optional auxiliary CNN-BiLSTM emotion inference
- confidence, clarity, and engagement scoring
- feedback generation
- JSON output saving
- transcription caching
- batch evaluation
- PyTest validation

## Pipeline Variants

The system can run in three modes:

| Variant | Acoustic metrics | Transcription/text metrics | Optional emotion | Use case |
|---|---:|---:|---:|---|
| `speech_only` | Yes | No | Yes | Acoustic-only ablation |
| `text_only` | No | Yes | No | Linguistic-only ablation |
| `multimodal` | Yes | Yes | Yes | Full system |

## Quick Start

Install dependencies in a virtual environment:

```bash
pip install -r requirements.txt
```

Run the full multimodal pipeline:

```bash
python -m app.main --file data/example.wav --variant multimodal
```

Run the other variants:

```bash
python -m app.main --file data/example.wav --variant speech_only
python -m app.main --file data/example.wav --variant text_only
```

Enable optional emotion analysis:

```bash
python -m app.main --file data/example.wav --variant multimodal --use_emotion
```

Enable transcription caching for a single run:

```bash
python -m app.main --file data/example.wav --variant multimodal --use_transcription_cache
```

## Batch Evaluation

Run all variants over `data/main_eval` and save JSON outputs:

```bash
python -m app.evaluation.run_batch --data_dir data/main_eval --variants all --out_csv outputs/main_eval/main_eval_results.csv --save_json
```

Batch evaluation uses the CrisperWhisper transcription cache by default. Disable it with:

```bash
python -m app.evaluation.run_batch --data_dir data/main_eval --variants all --no_transcription_cache
```

Compare system scores against human scores:

```bash
python scripts/evaluate_against_humans.py
```

Current audited human-score evaluation outputs are saved in:

- `outputs/main_eval/human_evaluation_overall.csv`
- `outputs/main_eval/human_evaluation_summary.csv`
- `outputs/main_eval/human_evaluation_report.md`

## Output JSON

Single-run and batch outputs are saved under:

```text
outputs/<dataset>/raw/<variant>/<session_id>_<variant_code>.json
```

Example:

```text
outputs/main_eval/raw/multimodal/S01_T1_M.json
```

The main output fields are:

- `meta`
- `transcript`
- `raw_transcript`
- `emotion_output`
- `scores`
- `speech_metrics`
- `text_metrics`
- `feedback`
- `latency_ms`
- `warnings`

The representative output contract is documented in:

```text
schema/output_schema.json
docs/system_io_and_output_schema.md
```

## Testing

Run the test suite:

```bash
pytest -v
```

Latest audited result:

```text
29 passed, 1 skipped
```

The skipped test is the optional real CrisperWhisper integration test. To run it:

```bash
RUN_CRISPERWHISPER_INTEGRATION=1 pytest -v tests/test_crisper_whisper.py
```

## Repository Structure

```text
app/
  audio/              audio loading, normalisation, silence trimming
  speech_analysis/    acoustic metrics and speech rate
  transcription/      CrisperWhisper adapter, result types, transcription cache
  text_analysis/      linguistic metrics
  emotion/            optional CNN-BiLSTM emotion predictor
  scoring/            confidence, clarity, engagement scoring
  feedback/           deterministic feedback generation
  output/             JSON result building and saving
  pipeline/           pipeline runner
  evaluation/         batch evaluation utilities

data/                 local audio files
docs/                 dissertation evidence and implementation notes
docs/figures/         generated Chapter 3 figures
outputs/              generated JSON, CSV, cache, and evaluation outputs
schema/               representative output schema
scripts/              evaluation/helper scripts
tests/                PyTest suite
```

## Dissertation Documentation

Key Chapter 3 evidence files:

- `docs/chapter3_system_architecture.md`
- `docs/chapter3_codebase_evidence.md`
- `docs/scoring_formulas_and_thresholds.md`
- `docs/acoustic_processing_implementation.md`
- `docs/linguistic_processing_implementation.md`
- `docs/caching_implementation.md`
- `docs/emotion_model_implementation_notes.md`
- `docs/system_io_and_output_schema.md`
- `docs/testing_summary.md`
- `docs/chapter3_test_case_table.md`
- `docs/performance_and_latency.md`
- `docs/implementation_challenges_and_solutions.md`
- `docs/chapter3_drafting_notes.md`
- `docs/consistency_audit.md`

Chapter 2 update notes:

```text
docs/chapter2_update_notes.md
```

Generated Chapter 3 figures:

- `docs/figures/figure_3_1_system_architecture.png`
- `docs/figures/figure_3_3_loss_curve.png`
- `docs/figures/figure_3_4_accuracy_curve.png`
- `docs/figures/figure_3_5_sample_scores.png`
- `docs/figures/figure_3_6_latency_breakdown.png`

## Important Limitations

- The confidence, clarity, and engagement formulas are interpretable heuristics, not objectively correct psychological measures.
- Scoring thresholds were refined using the available human-scored evaluation set, so generalisation to unseen speakers requires a separate held-out validation set.
- Emotion inference is optional and is not included in current `main_eval` outputs unless the batch is rerun with `--use_emotion`.
- CrisperWhisper punctuation is not treated as fully reliable; text metrics use ASR-safe chunking for readability-style features.
- The output schema is a representative contract, not a formally enforced JSON Schema validator.
