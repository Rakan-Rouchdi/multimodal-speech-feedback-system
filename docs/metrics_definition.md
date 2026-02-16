# Metrics and scoring definitions (v1)

## Headline scores (0–100)
- ConfidenceScore (0–100)
- ClarityScore (0–100)
- EngagementScore (0–100)

## Bands
- 0–39: Needs improvement
- 40–69: Developing
- 70–84: Strong
- 85–100: Excellent

## Speech metrics (units)
- speech_rate_wpm: words per minute
- pause_count: number of pauses detected
- mean_pause_sec: mean pause duration in seconds
- total_pause_sec: total pause time in seconds
- energy_mean: mean short-time energy proxy
- pitch_mean_hz: mean pitch estimate in Hz
- pitch_std_hz: pitch variation in Hz
- emotion: distribution over emotion labels

## Text metrics
- transcript: speech-to-text output
- word_count: number of words in transcript
- filler_count: number of filler words detected
- filler_rate_per_100w: (filler_count / word_count) * 100
- repeat_rate: proportion of repeated words/phrases (definition to be implemented)
- readability_proxy: simple readability score (definition to be implemented)
- sentiment: polarity [-1, 1] and subjectivity [0, 1] if available

## Output contract
The system returns a JSON object that conforms to schema/output_schema_v1.json.
Fields may be null depending on pipeline variant:
- speech_only: text_metrics may be null
- text_only: speech_metrics may be null
- multimodal: both present
