# Testing Summary

## How to Run

```bash
pytest -v
```

Last audited result:

```text
29 passed, 1 skipped
```

The skipped test is the optional real CrisperWhisper integration test, which requires:

```bash
RUN_CRISPERWHISPER_INTEGRATION=1 pytest -v tests/test_crisper_whisper.py
```

## Test Files

| Test file | Type | What it validates |
|---|---|---|
| `tests/test_audio_preprocessing.py` | Unit | Audio preprocessing returns waveform, sample rate, and duration. |
| `tests/test_speech_metrics.py` | Unit/edge | Silent and very short audio do not crash acoustic metrics. |
| `tests/test_speech_rate.py` | Unit | WPM formula. |
| `tests/test_text_metrics.py` | Unit/edge | Filler detection, filler rate, repetition rate, empty transcript handling, raw/clean transcript fields, ASR-safe clause splitting. |
| `tests/test_scoring.py` | Unit | Scores remain in `0-100`; banding categories. |
| `tests/test_scoring_bands.py` | Unit | Band boundary behaviour. |
| `tests/test_feedback_generation.py` | Unit | Low clarity triggers clarity feedback. |
| `tests/test_pipeline_runner.py` | Integration | Variants, invalid variant errors, logging fields, score presence, transcription cache use, JSON save. |
| `tests/test_transcription_cache.py` | Unit/integration | Cache write/hit, timestamp validation, implausible cache rejection, cleaning refresh. |
| `tests/test_crisper_whisper.py` | Unit/integration optional | Cleaning, adapter shape, word timestamps, fallback without VAD, optional real model run. |
| `tests/test_emotion_features.py` | Unit | Emotion feature extraction returns `(250, 53)`. |

## Unit Tests

Unit tests cover deterministic functions such as:

- text counting and repetition metrics
- score banding
- speech rate calculation
- audio preprocessing output shape
- emotion feature extraction shape

## Integration Tests

`tests/test_pipeline_runner.py::test_integration_pipeline_json_save` runs the pipeline with a stubbed transcriber and saves JSON.

`tests/test_crisper_whisper.py::test_real_crisperwhisper_pipeline_on_test_wav_without_cache` is an optional real-model integration test and is skipped unless the environment variable is set.

## Edge Case Tests

Implemented edge cases include:

- empty transcript
- silent audio
- very short audio
- invalid pipeline variant
- missing word timestamps in cache
- implausibly short cached transcription for long audio
- missing punctuation in transcript clause metrics

## Cache Tests

`tests/test_transcription_cache.py` proves:

- cache writes and subsequent hits
- invalid cache entries without word timestamps are rejected
- implausibly short long-audio cache entries are rejected
- old cleaning versions can be refreshed without retranscription

## Emotion Tests

The test suite validates feature extraction shape only. It does not run `app/VoiceModel/best_model.h5` during normal tests.

## Known Limitations

- The real CrisperWhisper model is not run by default in CI/local tests because it is slow and requires model availability.
- Full Keras emotion inference is not tested automatically.
- Tests validate robustness and deterministic behaviour, but not human perceptual correctness.
- Scoring formula generalisation requires additional held-out human-scored data.
