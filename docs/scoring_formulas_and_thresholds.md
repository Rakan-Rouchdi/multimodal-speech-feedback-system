# Scoring Formulas and Thresholds

Source file: `app/scoring/scoring.py`

## Score Range and Bands

All headline scores are clamped to `0-100` by:

```python
def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))
```

Band thresholds are defined in `app/contracts/constants.py`:

| Band | Inclusive score range |
|---|---:|
| Needs improvement | 0-39 |
| Developing | 40-69 |
| Strong | 70-84 |
| Excellent | 85-100 |

`band_for(score)` rounds the score, clamps to `0-100`, then checks the table above.

## Subscore Helper Functions

Range score:

```python
score_from_range(value, ideal_min, ideal_max, hard_min, hard_max)
```

- `100` inside `[ideal_min, ideal_max]`.
- `0` at or outside `hard_min`/`hard_max`.
- Linear falloff between hard and ideal bounds.

Lower-is-better score:

```python
score_lower_better(value, ideal_max, hard_max)
```

- `100` at or below `ideal_max`.
- `0` at or above `hard_max`.

Higher-is-better score:

```python
score_higher_better(value, hard_min, ideal_min)
```

- `100` at or above `ideal_min`.
- `0` at or below `hard_min`.

Missing evidence handling:

```python
coverage_adjusted_score(items, neutral=50.0)
```

The scorer averages only available subscores, then shrinks the result toward `50` according to available feature weight. This prevents single-modality runs from receiving extreme scores because missing features are ignored.

## Feature Thresholds

| Subscore | Function | Thresholds |
|---|---|---|
| `wpm_score` | `score_from_range` | ideal `120-165`, hard `85-210` WPM |
| `filler_score` | `score_lower_better` | ideal max `1.0`, hard max `6.0` fillers per 100 words |
| `repeat_score` | `score_lower_better` | ideal max `0.003`, hard max `0.035` adjacent repetition rate |
| `lexical_diversity_score` | `score_higher_better` | hard min `0.50`, ideal min `0.75` |
| `pitch_var_score` | `score_from_range` | ideal `10-25`, hard `3-65` Hz pitch standard deviation |
| `mean_pause_score` | `score_from_range` | ideal `0.12-0.45`, hard `0.05-1.20` seconds |
| `pause_rate_score` | `score_lower_better` | ideal max `12.0`, hard max `35.0` pauses per minute |
| `pause_ratio_score` | `score_lower_better` | ideal max `0.07`, hard max `0.16` |
| `energy_score` | `score_from_range` | ideal `0.035-0.12`, hard `0.003-0.18` RMS energy |

`pause_rate_score` is calculated and returned in `subscores`, but is not currently used in the headline score formulas.

## Weights

Confidence weights:

```python
CONFIDENCE_WEIGHTS = {
    "filler_score": 0.30,
    "repeat_score": 0.25,
    "energy_score": 0.12,
    "pitch_var_score": 0.05,
    "mean_pause_score": 0.08,
    "wpm_score": 0.08,
    "pause_ratio_score": 0.07,
    "lexical_diversity_score": 0.05,
    "emotion_confidence_score": 0.10,
}
```

Clarity weights:

```python
CLARITY_WEIGHTS = {
    "filler_score": 0.35,
    "repeat_score": 0.25,
    "energy_score": 0.12,
    "pause_ratio_score": 0.08,
    "mean_pause_score": 0.08,
    "wpm_score": 0.05,
    "lexical_diversity_score": 0.07,
}
```

Engagement weights:

```python
ENGAGEMENT_WEIGHTS = {
    "energy_score": 0.20,
    "pitch_var_score": 0.10,
    "filler_score": 0.25,
    "repeat_score": 0.20,
    "wpm_score": 0.08,
    "mean_pause_score": 0.04,
    "pause_ratio_score": 0.08,
    "lexical_diversity_score": 0.05,
    "emotion_engagement_score": 0.10,
}
```

## Exact Headline Formulas

The formulas are implemented as weighted item lists passed to `coverage_adjusted_score`.

Confidence:

```python
confidence_items = [
    (0.30, filler_score),
    (0.25, repeat_score),
    (0.12, energy_score),
    (0.05, pitch_var_score),
    (0.08, mean_pause_score),
    (0.08, wpm_score),
    (0.07, pause_ratio_score),
    (0.05, lexical_diversity_score),
]
```

If emotion is enabled and available, `(0.10, emotion_confidence_score)` is appended.

Clarity:

```python
clarity_items = [
    (0.35, filler_score),
    (0.25, repeat_score),
    (0.12, energy_score),
    (0.08, pause_ratio_score),
    (0.08, mean_pause_score),
    (0.05, wpm_score),
    (0.07, lexical_diversity_score),
]
```

Engagement:

```python
engagement_items = [
    (0.10, pitch_var_score),
    (0.20, energy_score),
    (0.25, filler_score),
    (0.20, repeat_score),
    (0.08, wpm_score),
    (0.04, mean_pause_score),
    (0.08, pause_ratio_score),
    (0.05, lexical_diversity_score),
]
```

If emotion is enabled and available, `(0.10, emotion_engagement_score)` is appended.

## Multimodal Consistency Penalty

When both speech and text evidence are available, poor textual fluency or excessive pause ratio applies a small penalty:

```python
fluency_penalty = 0.0
if filler_score is not None:
    fluency_penalty += 100.0 - filler_score
if repeat_score is not None:
    fluency_penalty += 3.0 * (100.0 - repeat_score)
if pause_ratio_score is not None:
    fluency_penalty += 100.0 - pause_ratio_score

confidence -= 0.02 * fluency_penalty
clarity -= 0.03 * fluency_penalty
engagement -= 0.04 * fluency_penalty
```

This penalty only applies when `speech_available` and `text_available` are both true, which corresponds to the multimodal case.

## Emotion Scoring

Emotion is optional. `run_pipeline(..., use_emotion=True)` attaches emotion data to `speech_metrics` for audio-enabled variants.

`EMOTION_SCORE_MAP` maps labels to `(confidence, engagement)` subscores:

| Label | Confidence | Engagement |
|---|---:|---:|
| neutral | 85 | 40 |
| calm | 95 | 35 |
| happy | 70 | 95 |
| surprised | 55 | 85 |
| angry | 40 | 60 |
| fearful | 25 | 30 |
| sad | 30 | 20 |
| disgust | 35 | 25 |

The code uses a probability-weighted blend over all labels, not only the top label.

## How Variants Differ

- `speech_only`: speech features are available; text features are absent. `coverage_adjusted_score` shrinks missing text evidence toward neutral.
- `text_only`: text features are available; acoustic metrics are absent. WPM can still be calculated from transcript word count and audio duration.
- `multimodal`: both speech and text features are available. The consistency penalty can apply.

## Dissertation Explanation

These formulas are heuristic and interpretable. The code does not implement a learned regression model for confidence, clarity, or engagement. The formulas combine measurable acoustic and linguistic proxies using explicit thresholds and weights.

Thresholds were refined through implementation testing and comparison with the available human-scored evaluation set (`outputs/main_eval/human_scores.csv`). The repository provides evidence of the final evaluation in `outputs/main_eval/human_evaluation_overall.csv`, where multimodal mean Spearman is `0.680229` and mean Pearson is `0.649114`.

Do not claim the thresholds are objectively correct or clinically validated. A safe dissertation statement is:

> The scoring system uses interpretable heuristic rules refined against the available human-score evaluation set. The rules are suitable for transparent system feedback, but further validation on a separate held-out dataset would be required before claiming generalisable human-level judgement.

Needs confirmation:

- If Chapter 2 claims a formal theory-driven derivation for each numeric threshold, that is not present in the code. Add supporting rationale or describe the thresholds as implementation-tuned heuristics.
