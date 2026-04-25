# System Input/Output and JSON Schema

## Single-File CLI Commands

```bash
python -m app.main --file data/example.wav --variant multimodal
python -m app.main --file data/example.wav --variant speech_only
python -m app.main --file data/example.wav --variant text_only
```

Optional flags:

```bash
--use_emotion
--use_transcription_cache
--transcription-cache-path outputs/cache/crisperwhisper_transcriptions.json
--output-dir outputs
--dataset <label>
```

## Batch Command

```bash
python -m app.evaluation.run_batch --data_dir data/main_eval --variants all --out_csv outputs/main_eval/main_eval_results.csv --save_json
```

`--variants` accepts `speech_only`, `text_only`, `multimodal`, or `all`.

## Input Expectations

Audio file extensions supported by batch discovery:

```python
{".wav", ".mp3", ".m4a", ".flac", ".ogg"}
```

Single-file CLI checks that the path exists before running.

## Output Location

`save_result_json()` writes:

```text
outputs/<dataset>/raw/<variant>/<session_id>_<variant_code>.json
```

Variant codes:

- `M`: multimodal
- `S`: speech_only
- `T`: text_only

Example:

```text
outputs/main_eval/raw/multimodal/S01_T1_M.json
```

## Output JSON Fields

Current output keys:

```text
meta
transcript
raw_transcript
emotion_output
scores
speech_metrics
text_metrics
feedback
latency_ms
warnings
```

Schema reference:

```text
schema/output_schema.json
```

The schema file is a representative contract, not an enforced JSON Schema validator.

## Required vs Optional Fields

| Field | Required | Notes |
|---|---:|---|
| `meta` | Yes | Contains file, duration, sample rate, variant, cache flags. |
| `transcript` | Yes | Empty string for `speech_only`. Cleaned transcript for text-enabled variants. |
| `raw_transcript` | Yes | Empty string for `speech_only`. Raw CrisperWhisper transcript for text-enabled variants. |
| `emotion_output` | Yes | `null` unless `--use_emotion` is enabled and variant is audio-enabled. |
| `scores` | Yes | Confidence, clarity, engagement, and bands. |
| `speech_metrics` | Variant-dependent | `null` for `text_only`. |
| `text_metrics` | Variant-dependent | `null` for `speech_only`. |
| `feedback` | Yes | Summary, bullets, next practice. |
| `latency_ms` | Yes | Stage timings in milliseconds. |
| `warnings` | Yes | Currently usually empty. |

## Example Output Snippet

From `outputs/main_eval/raw/multimodal/S01_T1_M.json`:

```json
{
  "meta": {
    "session_id": "S01_T1",
    "filename": "S01_T1.wav",
    "pipeline_variant": "multimodal",
    "transcription_source": "crisper_whisper",
    "transcription_cache_enabled": true,
    "transcription_cache_hit": true
  },
  "scores": {
    "confidence": 63.4,
    "clarity": 56.5,
    "engagement": 66.1,
    "bands": {
      "confidence": "Developing",
      "clarity": "Developing",
      "engagement": "Developing"
    }
  },
  "latency_ms": {
    "preprocess": 48.3,
    "transcription": 12.0,
    "speech_analysis": 6252.6,
    "emotion_analysis": 0.0,
    "text_analysis": 0.5,
    "fusion": 0.1,
    "feedback": 0.1,
    "total": 6313.6
  }
}
```

Values may change after reruns.

## Chapter 4 Support

`outputs/main_eval/main_eval_results.csv` flattens JSON results into rows for evaluation. `scripts/evaluate_against_humans.py` merges this CSV with `outputs/main_eval/human_scores.csv` and writes:

- `outputs/main_eval/final_merged.csv`
- `outputs/main_eval/human_evaluation_summary.csv`
- `outputs/main_eval/human_evaluation_overall.csv`
- `outputs/main_eval/human_evaluation_report.md`
