# Performance and Latency

## Timing Implementation

Timing is implemented by `app/utils/timing.py::Timer`:

```python
with timer.track("stage_name"):
    ...
```

The context manager stores elapsed milliseconds in `timer.ms`.

## Timed Stages

`app/pipeline/runner.py::run_pipeline` records:

- `preprocess`
- `transcription`
- `speech_analysis`
- `emotion_analysis`
- `text_analysis`
- `fusion`
- `feedback`
- `total`

The output field is:

```json
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
```

Example values above are from `outputs/main_eval/raw/multimodal/S01_T1_M.json` and may vary by hardware/run.

## Typical Timings From Current Outputs

The current outputs show that:

- cached transcription can be very fast because only cache retrieval is timed
- speech analysis can dominate runtime because pitch extraction with `librosa.pyin` is relatively expensive
- emotion analysis is `0.0` in current `main_eval` outputs because emotion was not enabled

## Caching Effect

Caching affects only the transcription stage. Batch evaluation uses transcription caching by default, so repeated runs avoid re-running CrisperWhisper for text-enabled variants when valid cached entries exist.

## Figure

Use:

```text
docs/figures/figure_3_6_latency_breakdown.png
```

The figure plots stage-wise latency from a real sample output JSON.

## Limitations

- Timings are local-machine dependent.
- Current timing does not separate cache lookup time from model transcription time in the same field.
- `total` is computed as the sum of recorded stage timings, not an outer wall-clock timer around the entire function.
- Existing outputs are not a controlled benchmark unless generated under fixed hardware and environment conditions.
