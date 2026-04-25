# Emotion Model Implementation Notes

This document records only details confirmed from the repository.

## Runtime Files

- `app/emotion/predictor.py`
- `app/VoiceModel/best_model.h5`
- `app/VoiceModel/Training.ipynb`
- `tests/test_emotion_features.py`

## Runtime Status

The emotion model is optional. It runs only when:

```bash
--use_emotion
```

is passed and the selected variant is `speech_only` or `multimodal`.

In current `outputs/main_eval` JSON files, `emotion_output` is `null` because the batch was run without `--use_emotion`.

## Labels

Confirmed runtime label order in `app/emotion/predictor.py`:

```python
EMOTION_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]
```

Notebook emotion mapping:

```python
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
```

## Dataset

Confirmed from notebook:

```python
DATASET_PATH = "Audio"
```

The repository contains actor-folder WAV files under `app/VoiceModel/Audio/Actor_*`.

Needs confirmation:

- Formal dataset name/citation. The filename pattern and labels are consistent with RAVDESS-style data, but the notebook does not explicitly write the dataset citation.

## Input Features

Confirmed in `app/emotion/predictor.py::extract_features` and `app/VoiceModel/Training.ipynb`:

- 13 MFCCs
- 13 delta MFCCs
- 13 delta-delta MFCCs
- 1 zero-crossing-rate feature
- 12 chroma features
- 1 RMS/RMSE energy feature

Total feature dimension:

```text
13 + 13 + 13 + 1 + 12 + 1 = 53
```

Runtime shape:

```text
(250, 53)
```

Model input shape after batch expansion:

```text
(1, 250, 53)
```

Test evidence:

```python
tests/test_emotion_features.py::test_emotion_feature_extraction_returns_model_shape
```

## Train/Validation/Test Split

Notebook output confirms:

```text
Train: (5765, 250, 53) (5765, 8)
Val  : (144, 250, 53) (144, 8)
Test : (144, 250, 53) (144, 8)
```

Notebook code confirms:

```python
train_test_split(..., test_size=0.1, stratify=y_encoded, random_state=SEED)
train_test_split(..., test_size=0.1111, stratify=np.argmax(y_train_val, axis=1), random_state=SEED)
```

Training set is augmented after splitting.

## Data Augmentation

Training notebook defines:

- `noise(data)`
- `stretch(data, rate=0.85)`
- `shift(data)`
- `pitch(data, sr, pitch_factor=0.7)`

Needs confirmation:

- Exact number of augmentations per original sample should be cited from notebook code if included in Chapter 3. The final augmented training shape is confirmed.

## Architecture

Confirmed from `app/VoiceModel/Training.ipynb`:

```python
model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),

    Dense(y_train_cat.shape[1], activation='softmax')
])
```

## Optimiser, Loss, Batch Size, Epochs

Confirmed:

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(..., epochs=50, batch_size=32, callbacks=[early_stop, checkpoint])
```

Callbacks:

```python
EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
```

Needs confirmation:

- Learning rate is not explicitly written in the notebook. If citing a value, either confirm Keras default Adam learning rate for the installed version or state that the default Adam settings were used.

## Saved Model Path

Notebook checkpoint filename:

```text
best_model.h5
```

Runtime path:

```python
MODEL_PATH = Path(__file__).resolve().parent.parent / "VoiceModel" / "best_model.h5"
```

Resolved repository path:

```text
app/VoiceModel/best_model.h5
```

Needs confirmation:

- The repository does not prove cryptographically that the current `app/VoiceModel/best_model.h5` is exactly the checkpoint from the logged notebook run.

## Metrics Confirmed From Notebook Output

Best validation checkpoint line:

```text
Epoch 26: val_accuracy improved from 0.77083 to 0.81944, saving model to best_model.h5
```

Early stopping:

```text
Epoch 36: early stopping
Restoring model weights from the end of the best epoch: 26.
```

Test classification report:

```text
accuracy 0.806 on 144 test samples
macro avg f1-score 0.806
weighted avg f1-score 0.804
```

## Runtime Prediction Output

`predict_emotion(file_path)` returns:

```python
{
    "top_label": top_label,
    "probabilities": {label: probability}
}
```

If feature extraction fails, it returns `neutral` with uniform probabilities.

## How Emotion Affects Scoring

In `app/scoring/scoring.py`, emotion probabilities are mapped to confidence and engagement subscores via `EMOTION_SCORE_MAP`. Emotion affects:

- confidence, if `emotion_confidence_score` is available
- engagement, if `emotion_engagement_score` is available

Emotion does not directly affect clarity in the current code.

## How Emotion Affects Feedback

`app/feedback/generator.py` adds emotion-aware feedback when `speech["emotion"]` contains a recognised `top_label`.

## Testing

Automated tests do not load the full Keras model. `tests/test_emotion_features.py` validates feature extraction shape `(250, 53)`.

## Figure Notes

- Figure 3.2 model architecture: use the layer list above.
- Figure 3.3 loss curve: generated at `docs/figures/figure_3_3_loss_curve.png` from notebook epoch logs.
- Figure 3.4 accuracy curve: generated at `docs/figures/figure_3_4_accuracy_curve.png` from notebook epoch logs.
