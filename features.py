import numpy as np
#import pandas as pd
import librosa


FEATURES = [
    'spectrogram',
    'mel_spectrogram',
    'spectral_centroid',
    'spectral_rolloff',
    'spectral_flux',
    'zero_crossings',
    'zero_crossings_rate',
    'chromogram'
]

def sort_features_list(lis):
    return sorted(lis, key=lambda f: FEATURES.index(f))

def spectrogram(buffer, sr=22050, fft_size=2048, hop_length=2048):
    # Amplitude spectrum (STFT)
    stft = librosa.stft(buffer, n_fft=fft_size, hop_length=hop_length)
    stft = np.abs(stft)
    # DB spectrum
    db_spect = librosa.amplitude_to_db(stft, ref=np.max)
    return np.mean(db_spect[0])

def mel_spectrogram(buffer, sr=22050, fft_size=2048, hop_length=2048):
    # Mel Spectrogram
    mel_spect = librosa.feature.melspectrogram(y=buffer, sr=sr)
    db_mel_spect = librosa.amplitude_to_db(mel_spect, ref=np.max)
    return np.mean(db_mel_spect[0])

def spectral_centroid(buffer, sr=22050, fft_size=2048, hop_length=2048):
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=buffer, sr=sr)[0]
    #spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    #spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)
    return np.mean(spectral_centroids)

def spectral_rolloff(buffer, sr=22050, fft_size=2048, hop_length=2048):
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=buffer, sr=sr)
    return np.mean(spectral_rolloff[0])

def spectral_flux(buffer, sr=22050, fft_size=2048, hop_length=2048):
    # Spectral flux
    spectral_flux = librosa.onset.onset_strength(y=buffer, sr=sr)
    return np.mean(spectral_flux)

def zero_crossings(buffer, sr=22050, fft_size=2048, hop_length=2048):
    # Zero crossings
    zero_crossings = librosa.zero_crossings(buffer, pad=False)
    return np.sum(zero_crossings)

def zero_crossings_rate(buffer, sr=22050, fft_size=2048, hop_length=2048):
    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(buffer)
    return np.mean(zero_crossing_rate[0])

def chromogram(buffer, sr=22050, fft_size=2048, hop_length=2048):
    # Chromagram
    chromogram = librosa.feature.chroma_stft(y=buffer, sr=sr, hop_length=hop_length)
    return [np.mean(chroma) for chroma in chromogram]

fns = locals()

def extract_features(filename, s, buffer, sr, features_list="all", fft_size=2048, hop_length=2048, *args, **kwargs):

    features_list = FEATURES if features_list=="all" else \
        sort_features_list(features_list)

    features = {
        "filename": filename,
        "timestamp": librosa.samples_to_time(s, sr=sr)
    }

    for fname in features_list:
        if not fname=="chromogram":
            features[fname] = fns[fname](buffer, sr=sr, fft_size=fft_size, hop_length=hop_length)
        else:
            chromas = chromogram(buffer, sr=sr, fft_size=fft_size, hop_length=hop_length)
            for i, chroma in enumerate(chromas, start=1):
                features[f"chroma{i}"] = chroma

    for (key, value) in features.items():
        if not isinstance(value, str):
            features[key] = value.item()

    return features

'''
import librosa
import pprint
from features import extract_features 
file_path = '../Dropbox/Miscellaneous/TAAT/Data/Original Test Data/data/17-Marimba-Strikes.wav' 
audio, sr = librosa.load(file_path, sr=22050, mono=True)
 sr 
feat = extract_features(file_path, 0, audio, sr, ["mel_spectrogram", "spectral_rolloff", "spectrogram"], 2048, 2048)
pprint.pp(feat)
'''
