# Emotion Model Implementation Notes

This note documents the auxiliary emotion model exactly as evidenced in the codebase and notebook outputs.

## Primary sources inspected
- [app/emotion/predictor.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/emotion/predictor.py)
- [app/VoiceModel/Training.ipynb](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/VoiceModel/Training.ipynb)
- [app/VoiceModel/best_model.h5](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/VoiceModel/best_model.h5)

## Purpose
The emotion model is optional. It produces an emotion distribution from audio and can be incorporated into the multimodal scoring stage when `--use_emotion` is enabled.

## Dataset used
- RAVDESS speech emotion audio dataset stored under [app/VoiceModel/Audio](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/VoiceModel/Audio)
- Eight emotion labels are used:
  - angry
  - calm
  - disgust
  - fearful
  - happy
  - neutral
  - sad
  - surprised

## Input features
The model operates on frame-level acoustic features, not raw waveform samples.

Feature stack evidenced in [app/emotion/predictor.py](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/emotion/predictor.py):
- 13 MFCC coefficients
- 13 delta coefficients
- 13 delta-delta coefficients
- 1 zero-crossing-rate feature
- 12 chroma features
- 1 RMS energy feature

Total input dimensionality: `53` features per frame.

Time dimension:
- each sample is padded or truncated to `250` time steps
- final model input shape is `(250, 53)`

## Data split and augmentation
Evidence from notebook cells in [app/VoiceModel/Training.ipynb](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/VoiceModel/Training.ipynb):
- original samples are extracted first
- split is stratified by emotion label
- first split: train/validation pool vs test with `test_size=0.1`
- second split: train vs validation with `test_size=0.1111`

Logged tensor shapes:
- training set after augmentation: `(5765, 250, 53)`
- validation set: `(144, 250, 53)`
- test set: `(144, 250, 53)`

Training augmentation is applied only to the training partition. The notebook includes:
- additive noise
- time stretching
- time shifting
- pitch shifting

## Model architecture
Evidence from model definition cells in [app/VoiceModel/Training.ipynb](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/VoiceModel/Training.ipynb):

1. `Conv1D(64, kernel_size=5, activation="relu", kernel_regularizer=l2(0.001))`
2. `BatchNormalization()`
3. `MaxPooling1D()`
4. `Dropout(0.3)`
5. `Conv1D(128, kernel_size=3, activation="relu", kernel_regularizer=l2(0.001))`
6. `BatchNormalization()`
7. `MaxPooling1D()`
8. `Dropout(0.3)`
9. `Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)))`
10. `Dropout(0.4)`
11. `Dense(128, activation="relu", kernel_regularizer=l2(0.001))`
12. `Dropout(0.3)`
13. `Dense(num_classes, activation="softmax")`

Summary description: CNN + Bidirectional LSTM classifier.

## Optimiser, loss, and training setup
Documented in the notebook:
- optimiser: `adam`
- loss function: `categorical_crossentropy`
- metric: `accuracy`
- batch size: `32`
- epochs requested: `50`

Callbacks:
- `EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)`
- `ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)`

## Saved model path
- training checkpoint filename in notebook: `best_model.h5`
- repository model path used at runtime: [app/VoiceModel/best_model.h5](/Users/rakanrouchdi/Desktop/speech-feedback-dissertation/app/VoiceModel/best_model.h5)

## Final metrics evidenced
Best validation checkpoint explicitly logged in the notebook:
- `Epoch 26: val_accuracy improved from 0.77083 to 0.81944, saving model to best_model.h5`

Reported held-out test result from the notebook output:
- test accuracy: `0.806` on `144` test samples

## Values that should not be guessed
The following values were not explicitly set in the inspected notebook cells and therefore should not be claimed as fixed implementation details:
- explicit learning rate value for Adam
- full confusion-matrix summary in dissertation prose unless copied directly from a saved notebook output
- claim that the current on-disk `best_model.h5` definitely came from one exact notebook run unless that provenance is separately verified
