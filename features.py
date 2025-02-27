import numpy as np
#import pandas as pd
import librosa


def extract_features(buffer, sr, fft_size, hop_length):

    # Amplitude spectrum (STFT)
    stft = librosa.stft(buffer, n_fft=fft_size, hop_length=hop_length)
    stft = np.abs(stft)

    # DB spectrum
    db_spect = librosa.amplitude_to_db(stft, ref=np.max)

    # Mel Spectrogram
    mel_spect = librosa.feature.melspectrogram(y=buffer, sr=sr)
    db_mel_spect = librosa.amplitude_to_db(mel_spect, ref=np.max)

    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=buffer, sr=sr)[0]
    #spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    #spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=buffer, sr=sr)[0]

    # Spectral flux
    spectral_flux = librosa.onset.onset_strength(y=buffer, sr=sr)

    # Zero crossings
    zero_crossings = librosa.zero_crossings(buffer, pad=False)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(buffer)

    # Chromagram
    chromagram = librosa.feature.chroma_stft(y=buffer, sr=sr, hop_length=hop_length)

    features = {
        "spectrum": np.mean(db_spect[0]),
        "mel_spectrogram": np.mean(db_mel_spect[0]),
        "spectral_centroid": np.mean(spectral_centroids),
        "spectral_rolloff": np.mean(spectral_rolloff),
        "spectral_flux": np.mean(spectral_flux),
        "zero_crossings": np.sum(zero_crossings),
        "zero_crossing_rate": np.mean(zero_crossing_rate[0]),
    }

    for (key, value) in features.items():
        features[key] = value.item()

    return features
