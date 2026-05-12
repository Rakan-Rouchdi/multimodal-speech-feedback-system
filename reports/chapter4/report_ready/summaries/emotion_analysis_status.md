# Emotion Analysis Status

The supplementary emotion analysis was generated for the 20 evaluation recordings in `data/main_eval/` using the existing runtime predictor, `app/emotion/predictor.py`, with model path `app/VoiceModel/best_model.h5`.

The predictor label mapping used was:

```python
["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
```

The predictor comments state that this label order must match the `LabelEncoder` used during training. The training evidence supports this mapping: `app/VoiceModel/Training.ipynb` imports `sklearn.preprocessing.LabelEncoder`, fits it to the RAVDESS emotion strings, transforms labels with `le.fit_transform(...)` / `le.transform(...)`, and saves `best_model.h5` via `ModelCheckpoint`. Scikit-learn `LabelEncoder` sorts string classes alphabetically, giving the same order as `predictor.py`.

The serialized `LabelEncoder` itself was not saved in the project files found for the CNN-BiLSTM `best_model.h5` workflow. The appendix outputs are therefore treated as supplementary and exploratory, with the mapping supported by the training notebook and predictor code rather than by a separate saved encoder artefact.

Generated appendix-only outputs:

- `appendix_emotion_predictions.csv` / `.md`
- `appendix_emotion_by_confidence_engagement_group.csv` / `.md`
- `appendix_emotion_correlation_summary.csv` / `.md`
- `appendix_emotion_distribution.png` / `.pdf`
- `appendix_emotion_by_confidence_group.png` / `.pdf`
- `appendix_emotion_by_engagement_group.png` / `.pdf`

Emotion labels are not used in the primary Chapter 4 MAE/Pearson/Spearman comparison. RAVDESS emotion labels are categorical acted-emotion classes and are not equivalent to the Chapter 4 confidence, clarity, or engagement dimensions.
