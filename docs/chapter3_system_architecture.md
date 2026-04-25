# Chapter 3 System Architecture

## Implemented Pipeline

The implemented system processes one audio file and produces a structured JSON result with scores and feedback. The main execution path is:

1. CLI call in `app/main.py`.
2. Pipeline orchestration in `app/pipeline/runner.py::run_pipeline`.
3. Audio preprocessing in `app/audio/preprocessing.py`.
4. Acoustic metrics in `app/speech_analysis/metrics.py`.
5. CrisperWhisper transcription in `app/transcription/crisper_whisper.py`.
6. Linguistic metrics in `app/text_analysis/metrics.py`.
7. Optional emotion prediction in `app/emotion/predictor.py`.
8. Multimodal scoring in `app/scoring/scoring.py`.
9. Feedback generation in `app/feedback/generator.py`.
10. JSON result construction and saving in `app/output/result_builder.py` and `app/output/save_json.py`.

## Main Modules

| Module | Responsibility |
|---|---|
| `app/main.py` | CLI entry point for a single pipeline run. |
| `app/pipeline/runner.py` | Coordinates preprocessing, enabled variant modules, scoring, feedback, timing, and output construction. |
| `app/audio/preprocessing.py` | Loads audio, resamples to 16 kHz, converts to mono, normalises amplitude, trims leading/trailing silence. |
| `app/speech_analysis/metrics.py` | Computes RMS energy, pause metrics, and pitch statistics. |
| `app/speech_analysis/speech_rate.py` | Computes words per minute from transcript word count and audio duration. |
| `app/transcription/crisper_whisper.py` | Loads `nyrahealth/faster_CrisperWhisper` through `faster_whisper.WhisperModel`, transcribes audio, returns segments and word timestamps. |
| `app/transcription/cache.py` | Optional transcription-result cache with model/version/timestamp validation. |
| `app/text_analysis/metrics.py` | Computes word counts, fillers, disfluencies, repetition rate, ASR-safe clause metrics, readability proxy, lexical diversity. |
| `app/emotion/predictor.py` | Optional Keras emotion model inference using `app/VoiceModel/best_model.h5`. |
| `app/scoring/scoring.py` | Converts metrics into confidence, clarity, engagement scores and bands. |
| `app/feedback/generator.py` | Maps score/metric conditions to deterministic feedback messages. |
| `app/output/result_builder.py` | Builds the final JSON dictionary. |
| `app/output/save_json.py` | Saves JSON outputs under `outputs/<dataset>/raw/<variant>/`. |

## Data Flow

`run_pipeline(file_path, variant, use_emotion, use_transcription_cache)` controls the data flow.

- `preprocess_audio()` returns `waveform`, `sample_rate`, and `duration_sec`.
- Audio-enabled variants call `compute_speech_metrics()` and derive `speech_rate_wpm`, `pause_rate_per_min`, and `pause_ratio`.
- Text-enabled variants call `CrisperWhisperTranscriber.transcribe()` directly or through `transcribe_with_cache()`.
- `compute_text_metrics(raw_transcript, clean_text)` computes linguistic metrics.
- The scorer receives `score_speech` and `score_text` dictionaries with explicit `available` flags.
- `scoring()` returns score values and subscores.
- `generate_feedback()` returns `summary`, `bullets`, and `next_practice`.
- `build_result()` creates the final output dictionary.

## CLI Entry Points

Single file:

```bash
python -m app.main --file data/example.wav --variant multimodal
python -m app.main --file data/example.wav --variant speech_only
python -m app.main --file data/example.wav --variant text_only
```

Optional emotion:

```bash
python -m app.main --file data/example.wav --variant multimodal --use_emotion
```

Optional transcription cache:

```bash
python -m app.main --file data/example.wav --variant multimodal --use_transcription_cache
```

Batch evaluation:

```bash
python -m app.evaluation.run_batch --data_dir data/main_eval --variants all --out_csv outputs/main_eval/main_eval_results.csv --save_json
```

Human-score evaluation:

```bash
python scripts/evaluate_against_humans.py
```

## Runtime Variants

| Variant | Speech metrics | Transcription/text metrics | Emotion possible | Notes |
|---|---:|---:|---:|---|
| `speech_only` | Yes | No | Yes, if `--use_emotion` | Text metrics are `null`; speech rate is `0.0` because no transcript is available. |
| `text_only` | No acoustic metrics | Yes | No | WPM is derived from transcript word count and audio duration. |
| `multimodal` | Yes | Yes | Yes, if `--use_emotion` | Uses both acoustic and text evidence, plus consistency penalty when text fluency/pause ratio are poor. |

## Caching

Caching is implemented only for CrisperWhisper transcription results in `app/transcription/cache.py`. Cache entries are stored by default at:

```text
outputs/cache/crisperwhisper_transcriptions.json
```

The cache key is:

```python
f"{Path(file_path).name}:{audio_sha256(file_path)}"
```

Caching is disabled by default for `app.main` and enabled by default in `app.evaluation.run_batch` unless `--no_transcription_cache` is passed.

## Timing and Latency

Timing is implemented in `app/utils/timing.py::Timer`. `run_pipeline()` records:

- `preprocess`
- `transcription`
- `speech_analysis`
- `emotion_analysis`
- `text_analysis`
- `fusion`
- `feedback`
- `total`

These values are written to the JSON field `latency_ms`.

## Optional Components

The auxiliary emotion model is optional. It is only run when `use_emotion=True` and the selected variant includes audio (`speech_only` or `multimodal`). The model path used at runtime is:

```text
app/VoiceModel/best_model.h5
```

The automated test suite validates emotion feature extraction shape, not full Keras model inference.

## Limitations and Assumptions

- The scoring system is heuristic and human-refined; it is not an objectively correct psychological measure.
- Human-score alignment was evaluated on `outputs/main_eval/human_scores.csv`; generalisation to unseen speakers requires a separate held-out set.
- CrisperWhisper punctuation is not fully reliable; text metrics therefore use ASR-safe chunking for clause/readability features.
- Emotion analysis is optional and not used in the current `main_eval` outputs unless the batch is rerun with `--use_emotion`.
- Full emotion model provenance needs confirmation if claiming the exact on-disk `best_model.h5` came from one specific notebook run.

## Suggested Figure 3.1

Use `docs/figures/figure_3_1_system_architecture.png`.

Title: `Implemented Multimodal Speech Feedback System Architecture`

Boxes and arrows:

```text
Audio Input -> Audio Preprocessing
Audio Preprocessing -> Acoustic Features
Audio Preprocessing -> CrisperWhisper Transcription
Audio Preprocessing -> Optional Emotion Model
CrisperWhisper Transcription -> Text Features
Acoustic Features -> Multimodal Fusion
Text Features -> Multimodal Fusion
Optional Emotion Model -> Multimodal Fusion
Multimodal Fusion -> Scoring + Bands
Scoring + Bands -> Feedback + JSON Output
```
