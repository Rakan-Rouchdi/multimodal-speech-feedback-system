# Metrics and scoring definitions

## Headline scores (0-100)
- ConfidenceScore (0-100)
- ClarityScore (0-100)
- EngagementScore (0-100)

These scores are produced in [app/scoring/scoring.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/scoring/scoring.py).
The implementation uses weighted evidence from the available modalities and
shrinks partial-modality scores toward a neutral midpoint rather than
over-rewarding missing evidence.

## Bands
- 0-39: Needs improvement
- 40-69: Developing
- 70-84: Strong
- 85-100: Excellent

## Speech metrics (units)
- speech_rate_wpm: words per minute
- pause_count: number of pauses detected
- mean_pause_sec: mean pause duration in seconds
- total_pause_sec: total pause time in seconds
- pause_rate_per_min: pause_count normalised by speech duration
- pause_ratio: total_pause_sec / duration_sec
- energy_mean: mean short-time energy proxy
- pitch_mean_hz: mean pitch estimate in Hz
- pitch_std_hz: pitch variation in Hz
- emotion: optional output from the auxiliary CNN-BiLSTM emotion model

## Text metrics
- transcript: speech-to-text output
- raw_word_count: number of words in the raw transcript
- clean_word_count: number of words in the cleaned transcript
- filler_count: number of filler words and bracketed disfluency markers detected
- disfluency_count: explicit CrisperWhisper disfluency markers such as `[UH]`, `[UM]`
- filler_rate_per_100w: `(filler_count / clean_word_count) * 100`
- repeat_rate: adjacent repetition rate on cleaned text, excluding filler-only repeats
- readability_proxy: 0-100 spoken readability proxy based on average clause length
- avg_clause_length: average number of words in punctuation-aware ASR-safe chunks; very long chunks are split into 20-word windows when punctuation is missing
- estimated_clause_count: number of punctuation-aware ASR-safe chunks
- lexical_diversity: unique cleaned words divided by total cleaned words

## Scoring formulas
- Confidence: weighted combination of filler score, repetition score, energy, controlled pitch variation, pause timing, speech-rate score, lexical diversity, and optional emotion-confidence score.
- Clarity: weighted combination of filler score, repetition score, energy, pause ratio, mean pause duration, speech-rate score, and lexical diversity.
- Engagement: weighted combination of energy, controlled pitch variation, filler score, repetition score, speech-rate score, pause timing, lexical diversity, and optional emotion-engagement score.
- Multimodal scoring applies a small consistency penalty when strong acoustic delivery is paired with poor textual fluency or excessive pause ratio.

The exact feature weights and band thresholds are defined in
[app/scoring/scoring.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/scoring/scoring.py)
and [app/contracts/constants.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/contracts/constants.py).

## Output contract
The system returns a JSON object that matches the representative contract in schema/output_schema.json.
Fields may be null depending on pipeline variant:
- speech_only: text_metrics may be null
- text_only: speech_metrics may be null
- multimodal: both present
