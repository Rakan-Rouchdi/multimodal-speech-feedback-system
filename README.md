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

| Variant         | Acoustic metrics | Transcription/text metrics | Optional emotion | Use case                 |
| --------------- | ---------------: | -------------------------: | ---------------: | ------------------------ |
| `speech_only` |              Yes |                         No |              Yes | Acoustic-only ablation   |
| `text_only`   |               No |                        Yes |               No | Linguistic-only ablation |
| `multimodal`  |              Yes |                        Yes |              Yes | Full system              |

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
