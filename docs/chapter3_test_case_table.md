# Chapter 3 Test Case Table

| Test ID | Test type | Component | Input / scenario | Expected output | Actual result | Pass/Fail | Evidence file |
|---|---|---|---|---|---|---|---|
| T01 | Unit | Audio preprocessing | Synthetic speech WAV | Waveform returned, `sr == 16000`, duration > 0 | Passed in pytest | Pass | `tests/test_audio_preprocessing.py` |
| T02 | Unit/edge | Acoustic metrics | Silent WAV | Acoustic module does not crash; metrics returned | Passed in pytest | Pass | `tests/test_speech_metrics.py` |
| T03 | Unit/edge | Acoustic metrics | Very short WAV | Acoustic module does not crash | Passed in pytest | Pass | `tests/test_speech_metrics.py` |
| T04 | Unit | Speech rate | `word_count=150`, `duration=60` | WPM calculation correct | Passed in pytest | Pass | `tests/test_speech_rate.py` |
| T05 | Unit | Text metrics | `"Um I think you know this is like, kind of tricky."` | Filler count is 4 | Passed in pytest | Pass | `tests/test_text_metrics.py` |
| T06 | Unit | Text metrics | Adjacent repeated words | Repetition rate > 0 for repeated text, 0 for non-repeated text | Passed in pytest | Pass | `tests/test_text_metrics.py` |
| T07 | Edge | Text metrics | Empty transcript | Word counts and filler rate are 0; no crash | Passed in pytest | Pass | `tests/test_text_metrics.py` |
| T08 | Edge | Text metrics | 45 words with no full stops | ASR-safe average clause length is 15.0 | Passed in pytest | Pass | `tests/test_text_metrics.py` |
| T09 | Unit | Scoring | Valid speech/text score inputs | Confidence, clarity, engagement are within 0-100 | Passed in pytest | Pass | `tests/test_scoring.py` |
| T10 | Unit | Bands | Boundary scores 0, 39, 40, 69, 70, 84, 85, 100 | Expected category boundaries | Passed in pytest | Pass | `tests/test_scoring_bands.py` |
| T11 | Unit | Feedback | Low clarity score | Clarity-specific feedback present | Passed in pytest | Pass | `tests/test_feedback_generation.py` |
| T12 | Integration | Pipeline variants | Stub transcriber with all variants | Correct modules enabled/disabled | Passed in pytest | Pass | `tests/test_pipeline_runner.py` |
| T13 | Edge | Pipeline | Invalid variant | Raises clear `ValueError` | Passed in pytest | Pass | `tests/test_pipeline_runner.py` |
| T14 | Integration | Output JSON | Pipeline result saved to temp directory | JSON file exists; scores and feedback present | Passed in pytest | Pass | `tests/test_pipeline_runner.py` |
| T15 | Cache | Transcription cache | First and second cached calls | First miss, second hit | Passed in pytest | Pass | `tests/test_transcription_cache.py` |
| T16 | Cache edge | Missing word timestamps | Invalid cache entry | Entry rejected and refreshed | Passed in pytest | Pass | `tests/test_transcription_cache.py` |
| T17 | Cache edge | Long audio with too few words | Implausible cache entry | Entry rejected | Passed in pytest | Pass | `tests/test_transcription_cache.py` |
| T18 | Transcription | Fake CrisperWhisper response | Segments and word timestamps | Adapter returns expected structures | Passed in pytest | Pass | `tests/test_crisper_whisper.py` |
| T19 | Transcription edge | Long incomplete transcript with VAD | Retry without VAD | Fallback called and more words returned | Passed in pytest | Pass | `tests/test_crisper_whisper.py` |
| T20 | Emotion | Synthetic waveform | Feature shape `(250, 53)` | Passed in pytest | Pass | `tests/test_emotion_features.py` |
| T21 | Optional integration | Real CrisperWhisper on `data/test.wav` | Real pipeline returns transcript and scores | Skipped unless `RUN_CRISPERWHISPER_INTEGRATION=1` | Skipped | `tests/test_crisper_whisper.py` |
