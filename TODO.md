# Speech-Feedback-Dissertation Refactor TODO

## Current Status: [ ] Not started

Breakdown of approved plan into atomic steps. Mark [x] as completed.

### Phase 1: Transcription & Cache Fixes (Steps 1-3)

- [ ] **1a.** Edit `app/transcription/whisper_transcribe.py`: Add `clean_text: str`, `words: List[dict]` to TranscriptionResult dataclass
- [ ] **1b.** Edit `app/transcription/whisper_transcribe.py`: Update `word_count()` to prefer clean_text if available
- [ ] **2.** Edit `app/transcription/crisper_whisper.py`: Add `clean_transcript()` function; compute `clean_text`, store `words` in return
- [ ] **3a.** Edit `app/transcription/cache.py`: Add constants `CURRENT_MODEL_ID`, `CLEANING_VERSION`; staleness check in `get_or_transcribe()`
- [ ] **3b.** Edit `app/transcription/cache.py`: Update serialization to include `clean_text`, `words`, metadata

### Phase 2: Text Metrics & Scoring (Steps 4-5)

- [ ] **4.** Edit `app/text_analysis/metrics.py`: Dual-input `compute_text_metrics(transcript, clean_text)`; add raw/clean word counts
- [ ] **5a.** Edit `app/scoring/scoring_v1.py`: Implement asymmetric weights (confidence 0.65T/0.35S, clarity 0.70T/0.30S, engagement 0.15T/0.85S)
- [ ] **5b.** Edit `app/scoring/scoring_v1.py`: Text-only engagement → 50.0 neutral; emotion opt-in via param

### Phase 3: Pipeline & Runners (Steps 6-7)

- [ ] **6.** Edit `app/pipeline/runner.py`: Add `use_emotion=False`; pass clean_text to metrics; fix WPM; add transcription_source
- [ ] **7a.** Edit `app/evaluation/run_batch.py`: Default data_dir="data/main_eval"; --use_emotion flag
- [ ] **7b.** Edit `app/evaluation/run_batch.py`: Add new columns to CSV (filler_count etc.)
- [ ] **8.** Edit `app/main.py`: Add --use_emotion flag, pass to runner

### Phase 4: Tests & Cleanup

- [ ] Update tests/test_text_metrics.py, tests/test_pipeline_output.py for new fields
- [ ] rm -rf app/CrisperWhisper/recordings/* (remove duplicates)
- [ ] **Verification:**| Command                                                          | Expected                               |
  | ---------------------------------------------------------------- | -------------------------------------- |
  | python -m app.main data/main_eval/S01_T1.wav --variant text_only | readability>0, WC~160, engagement=50.0 |
  | python -m app.evaluation.run_batch --data_dir data/main_eval     | new columns, no 43.5 constant          |
  | pytest tests/                                                    | all pass                               |
  | cat app/transcription/transcription_cache.json\| jq .S01_T1      | has clean_text, model_id               |

**Next step:** Phase 1a - Edit TranscriptionResult dataclass
