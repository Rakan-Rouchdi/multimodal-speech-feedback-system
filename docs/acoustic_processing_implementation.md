# Acoustic Processing Implementation

## Files

- `app/audio/preprocessing.py`
- `app/speech_analysis/metrics.py`
- `app/speech_analysis/speech_rate.py`
- `app/pipeline/runner.py`

## Audio Loading

`app/audio/preprocessing.py::load_audio` uses:

```python
librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
```

Constants:

```python
TARGET_SAMPLE_RATE = 16000
```

This means input audio is resampled to 16 kHz and converted to mono by Librosa.

## Normalisation

`normalise_audio(waveform)` divides by the maximum absolute amplitude when the maximum is greater than zero:

```python
max_val = np.max(np.abs(waveform))
if max_val > 0:
    waveform = waveform / max_val
```

Silent audio is left unchanged because `max_val` is zero.

## Silence Trimming

`trim_silence(waveform, sr)` uses:

```python
librosa.effects.trim(waveform, top_db=20)
```

Only leading/trailing silence is removed during preprocessing.

## Energy/RMS

`app/speech_analysis/metrics.py::energy_mean` computes frame-level RMS:

```python
rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
energy_mean = np.mean(rms)
```

Empty waveform returns `0.0`.

## Pause Detection

`pause_metrics(waveform, sr, top_db=30, min_pause_sec=0.15)` uses:

```python
intervals = librosa.effects.split(waveform, top_db=top_db)
```

The implementation counts internal gaps only:

```python
for (_, end_a), (start_b, _) in zip(intervals[:-1], intervals[1:]):
    gap_sec = (start_b - end_a) / sr
    if gap_sec >= min_pause_sec:
        pause_durations.append(gap_sec)
```

Returned metrics:

- `pause_count`
- `mean_pause_sec`
- `total_pause_sec`

Derived in `app/pipeline/runner.py`:

```python
pause_rate_per_min = (pause_count / duration) * 60.0
pause_ratio = total_pause_sec / duration
```

## Pitch Extraction

`pitch_metrics(waveform, sr)` uses `librosa.pyin`:

```python
f0, voiced_flag, voiced_prob = librosa.pyin(
    waveform,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=2048,
    hop_length=512,
)
```

NaN/unvoiced frames are ignored. Returned metrics:

- `pitch_mean_hz`
- `pitch_std_hz`

If no pitch is available, both values are `0.0`.

## Speech Rate

`speech_rate_wpm(word_count, duration_sec)` uses:

```python
return (word_count / duration_sec) * 60.0
```

If `duration_sec <= 0`, it returns `0.0`.

In `text_only` and `multimodal`, `word_count` comes from `text_metrics["clean_word_count"]`. In `speech_only`, no transcript is available, so WPM is `0.0`.

## Libraries

- `librosa.load`
- `librosa.effects.trim`
- `librosa.effects.split`
- `librosa.feature.rms`
- `librosa.pyin`
- `numpy`

## Known Limitations

- Pitch extraction with `librosa.pyin` can be slow and may fail on noisy or very short audio.
- Pause detection depends on amplitude-based non-silent intervals and may be sensitive to background noise.
- Silence trimming happens before duration is calculated, so reported duration is post-trim duration.
- Speech rate depends on transcription availability and quality.
- Silent and very short audio are tested for non-crashing behaviour, but not for perceptual accuracy.

## Suggested Table

| Acoustic feature | Code source | Formula/tool | Output field |
|---|---|---|---|
| RMS energy | `energy_mean` | Mean Librosa RMS | `energy_mean` |
| Pauses | `pause_metrics` | Internal gaps from `librosa.effects.split` | `pause_count`, `mean_pause_sec`, `total_pause_sec` |
| Pitch | `pitch_metrics` | `librosa.pyin` voiced F0 | `pitch_mean_hz`, `pitch_std_hz` |
| Speech rate | `speech_rate_wpm` | `(word_count / duration) * 60` | `speech_rate_wpm` |
| Pause ratio | `run_pipeline` | `total_pause_sec / duration` | `pause_ratio` |
