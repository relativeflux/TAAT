import os
import sklearn
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from features import extract_features, features_to_dataframe


def get_kmeans_clusters(df, k):
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(df)
    clusters = kmeans.predict(df)
    return [(filename, clusters[i]) for (i, filename) in enumerate(df.index)]

def get_cluster(result, index):
    return list(filter(lambda x: x[1]==index, result))

FEATURES = [
    "mel_spectrogram",
    "spectral_centroid",
    "spectral_flux",
    "zero_crossings",
    "zero_crossings_rate"
]

def get_kmeans_dataframe(input_dir, features=FEATURES, sr=16000, fft_size=2048, hop_length=1024):
    feat = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                filepath = os.path.join(dirpath, filename)
                audio, _ = librosa.load(filepath, sr=sr, mono=True)
                feat.append(extract_features(filepath, 0, audio, sr, features, fft_size, hop_length))
    df = features_to_dataframe(feat)
    df.set_index( ['filename'], inplace=True)
    df.drop(['timestamp'], axis=1, inplace=True)
    return df

def get_kmeans_dataframe2(input_dir, sr=16000):
    feat = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                filepath = os.path.join(dirpath, filename)
                audio, _ = librosa.load(filepath, sr=sr, mono=True)
                mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=1024)
                db_mel_spect = librosa.amplitude_to_db(mel_spect, ref=np.max)
                mel_spect_bands_avg = np.mean(db_mel_spect, axis=1)
                feat_dict = {'filename': filepath}
                for (i, band) in enumerate(mel_spect_bands_avg):
                    feat_dict[f'band_{i}'] = band
                feat.append(feat_dict)
    df = features_to_dataframe(feat)
    df.set_index( ['filename'], inplace=True)
    return df

def plot_elbow(df, r=[2, 10], figsize=[6,4]):
    ssd = []
    for k in range(r[0], r[1]):
        kmeans = sklearn.cluster.KMeans(n_clusters=k)
        kmeans.fit(df)
        ssd.append(kmeans.inertia_)
    plt.figure(figsize=figsize, dpi=100)
    plt.plot(range(r[0], r[1]), ssd, color="green", marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("SSD for K")
    plt.show()
