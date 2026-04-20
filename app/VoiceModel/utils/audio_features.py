import numpy as np
import librosa

def extract_features_from_file(path, n_mfcc=13, top_db=40):
    try:
        y, sr = librosa.load(path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        delta_mean = np.mean(delta.T, axis=0)
        delta2_mean = np.mean(delta2.T, axis=0)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y_trimmed)
        zcr_mean = np.mean(zcr)

        # RMSE
        rmse = librosa.feature.rms(y=y_trimmed)
        rmse_mean = np.mean(rmse)

        # Duration
        duration = librosa.get_duration(y=y_trimmed, sr=sr)

        # Combine all features into a single vector
        feature_vector = np.concatenate([
            mfcc_mean,
            delta_mean,
            delta2_mean,
            chroma_mean,
            [zcr_mean, rmse_mean, duration]
        ])

        return feature_vector

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None
