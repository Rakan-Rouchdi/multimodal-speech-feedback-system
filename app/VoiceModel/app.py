import io
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


@st.cache_resource
def get_model():
    return load_model("best_model.h5", compile=False)

# Feature extraction function 
def extract_features(y, sr, max_len=250):
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=40)
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        zcr = librosa.feature.zero_crossing_rate(y_trimmed)
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        rmse = librosa.feature.rms(y=y_trimmed)
        features = np.vstack([mfcc, delta, delta2, zcr, chroma, rmse]).T
        if features.shape[0] < max_len:
            pad_width = max_len - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:max_len, :]
        return features
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def load_audio(uploaded_audio):
    audio_bytes = uploaded_audio.getvalue()
    return librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)


def predict_emotion(uploaded_audio):
    y, sr = load_audio(uploaded_audio)
    features = extract_features(y, sr)
    if features is None:
        return "Error"
    features = np.expand_dims(features, axis=0)
    prediction = get_model().predict(features, verbose=0)
    return EMOTION_LABELS[int(np.argmax(prediction))]

# Streamlit UI
st.title("Real-Time Emotion Recognition")
st.write("Record your voice or upload a WAV file, then let the model detect your emotion.")

recorded_audio = st.audio_input("Record your voice")
uploaded_audio = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

audio_source = recorded_audio or uploaded_audio

if audio_source:
    st.audio(audio_source)

    if st.button("Predict emotion"):
        with st.spinner("Listening closely..."):
            emotion = predict_emotion(audio_source)
        st.markdown(f"### Predicted Emotion: **{emotion.upper()}**")
