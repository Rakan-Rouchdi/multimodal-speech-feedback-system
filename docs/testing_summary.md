# Testing Summary

## How to run the tests

```bash
pytest -v
```

## Test files and what they validate
- [tests/test_audio_preprocessing.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_audio_preprocessing.py)
  - verifies the preprocessing pipeline returns waveform, sample rate, and duration for a valid WAV input

- [tests/test_speech_metrics.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_speech_metrics.py)
  - verifies the acoustic module handles silent and very short audio without crashing

- [tests/test_text_metrics.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_text_metrics.py)
  - verifies filler detection
  - verifies repetition-rate behaviour on adjacent repeated words
  - verifies empty transcript handling

- [tests/test_scoring.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_scoring.py)
  - verifies scoring outputs remain in the `0-100` range
  - verifies score band mapping

- [tests/test_scoring_bands.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_scoring_bands.py)
  - verifies the band boundaries at the threshold values

- [tests/test_feedback_generation.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_feedback_generation.py)
  - verifies low-clarity performance produces clarity-oriented feedback

- [tests/test_speech_rate.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_speech_rate.py)
  - verifies the words-per-minute calculation

- [tests/test_pipeline_runner.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_pipeline_runner.py)
  - verifies `speech_only`, `text_only`, and `multimodal` variants enable the expected modules
  - verifies invalid variants fail with a clear error
  - verifies pipeline results contain required scores, logging fields, and JSON save support
  - includes an integration-style test for a full pipeline run on a synthetic local WAV fixture

## Test categories

### Unit tests
- preprocessing
- text metrics
- speech metrics
- scoring
- feedback generation
- speech-rate helper

### Integration tests
- the integration-marked test in [tests/test_pipeline_runner.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/tests/test_pipeline_runner.py) runs the pipeline end to end on a small synthetic WAV file and verifies result structure and JSON output saving

### Edge-case tests
- empty transcript
- silent audio
- very short audio
- invalid pipeline variant

## Known limitations
- The integration test stubs transcription rather than loading the full CrisperWhisper model, which keeps local validation fast and deterministic.
- The optional CNN-BiLSTM emotion model is not exercised in full during automated tests because model loading is comparatively heavy; the pipeline test uses a stubbed emotion output when needed.
- Acoustic metrics on synthetic WAV fixtures confirm robustness and output structure, but they should not be cited as evidence of real-world model quality.
