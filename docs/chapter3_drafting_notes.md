# Chapter 3 Drafting Notes

## 3.1 Overview of Implementation

Write: The project implements a local Python pipeline for multimodal speech feedback, producing confidence, clarity, and engagement scores plus feedback from audio.

Evidence: `app/main.py`, `app/pipeline/runner.py`, `schema/output_schema.json`.

Do not claim: cloud deployment or real-time streaming.

## 3.2 System Architecture and Implementation Overview

Write: Describe the pipeline from audio input to preprocessing, acoustic/text/emotion branches, scoring, feedback, and JSON output.

Evidence: `docs/chapter3_system_architecture.md`, `docs/figures/figure_3_1_system_architecture.png`.

Figure: Figure 3.1 system architecture.

## 3.3 Development Environment and Tooling

Write: Python implementation using Librosa, faster-whisper, TensorFlow/Keras, pandas/scipy evaluation scripts, PyTest.

Evidence: `requirements.txt`, `pytest.ini`.

Do not claim: exact Python version unless reporting local test environment separately.

## 3.4 Auxiliary Emotion Model (CNN-BiLSTM)

Write: Optional model integrated through `app/emotion/predictor.py`; not required for normal pipeline runs.

Evidence: `docs/emotion_model_implementation_notes.md`.

Do not claim: emotion was used in current `main_eval` outputs unless rerun with `--use_emotion`.

### 3.4.1 Data Preparation

Write: Notebook uses `Audio` dataset path, emotion code mapping, feature extraction to `(250, 53)`, training augmentation.

Needs confirmation: formal dataset citation/name.

### 3.4.2 Network Architecture

Write: Conv1D -> BatchNorm -> MaxPool -> Dropout -> Conv1D -> BatchNorm -> MaxPool -> Dropout -> BiLSTM -> Dense -> Softmax.

Figure: Suggested Figure 3.2 architecture.

### 3.4.3 Training Configuration

Write: Adam, categorical crossentropy, accuracy, 50 epochs max, batch size 32, early stopping, checkpointing.

Needs confirmation: explicit learning rate if you want to state one.

### 3.4.4 Model Validation

Write: Notebook logs best validation accuracy 0.81944 at epoch 26 and test accuracy 0.806 on 144 samples.

Figures: `figure_3_3_loss_curve.png`, `figure_3_4_accuracy_curve.png`.

## 3.5 Acoustic Processing Module

Write: 16 kHz mono loading, normalisation, silence trimming, RMS energy, pause metrics, pitch metrics, WPM.

Evidence: `docs/acoustic_processing_implementation.md`.

Snippet: `librosa.load(..., sr=16000, mono=True)`.

## 3.6 Linguistic Processing Module

Write: CrisperWhisper transcription, raw/clean transcript, timestamps, cache, text metrics.

Evidence: `docs/linguistic_processing_implementation.md`, `docs/caching_implementation.md`.

### 3.6.1 Transcription

Write: `nyrahealth/faster_CrisperWhisper`, `word_timestamps=True`, VAD retry.

### 3.6.2 Text Feature Extraction

Write: fillers, disfluencies, filler rate, repetition, ASR-safe clause length, readability proxy, lexical diversity.

Do not claim: grammatical sentence parsing.

## 3.7 Multimodal Fusion and Scoring System

Write: Heuristic weighted scoring with missing-modality coverage adjustment and multimodal consistency penalty.

Evidence: `docs/scoring_formulas_and_thresholds.md`.

### 3.7.1 Scoring Logic and Thresholds

Include: weights and thresholds table from scoring documentation.

Do not claim: objective psychological measurement.

### 3.7.2 Adaptive Feedback Generation

Write: Deterministic rules generate summary, bullets, and practice tasks based on scores and metrics.

Evidence: `app/feedback/generator.py`.

## 3.8 System Variants Implementation

Write: `speech_only`, `text_only`, `multimodal`; explain enabled modules for each.

Evidence: `run_pipeline`, `tests/test_pipeline_runner.py`.

## 3.9 System Input/Output and Execution

Write: CLI commands, output JSON, latency fields, batch runner.

Evidence: `docs/system_io_and_output_schema.md`, `schema/output_schema.json`.

Figures: `figure_3_5_sample_scores.png`, `figure_3_6_latency_breakdown.png`.

## 3.10 System Validation and Testing

Write: PyTest suite plus human-score evaluation.

Evidence: `docs/testing_summary.md`, `docs/chapter3_test_case_table.md`, `outputs/main_eval/human_evaluation_overall.csv`.

### 3.10.1 Unit Testing

Use tests for text metrics, scoring, speech rate, audio preprocessing.

### 3.10.2 Integration Testing

Use pipeline JSON save and cache tests.

### 3.10.3 End-to-End Testing

Use batch command and generated `main_eval_results.csv`.

### 3.10.4 Robustness and Edge Case Handling

Use silent/short audio, empty transcript, invalid variant, incomplete cache entries.

### 3.10.5 Manual Validation

Use human-score evaluation. Current multimodal mean Spearman `0.680229`, Pearson `0.649114`, MAE `0.601933`.

Do not claim: held-out generalisation.

## 3.11 Implementation Challenges and Solutions

Use `docs/implementation_challenges_and_solutions.md`.

## 3.12 Implementation Summary

Write: Summarise that the system implements a complete auditable pipeline with variants, structured outputs, validation tests, and documented limitations.

Do not claim: production-grade deployment, clinical assessment, or universal validity.
