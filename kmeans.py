import os
import sklearn
import numpy as np
import pandas as pd
import librosa
from features import extract_features, features_to_dataframe


def kmeans_cluster(data, k):
    kmeans = sklearn.cluster.KMeans(n_clusters= k)
    return kmeans.fit(data)

def get_kmeans_clusters(input_dir, sr=16000, k=5):
    feat = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                filepath = os.path.join(dirpath, filename)
                audio, _ = librosa.load(filepath, sr=sr, mono=True)
                feature_list = ["mel_spectrogram", "spectral_centroid", "spectral_flux"]
                feat.append(extract_features(filepath, 0, audio, sr, feature_list, 2048, 2048))
    df = features_to_dataframe(feat)
    df.set_index( ['filename'], inplace=True)
    df.drop(['timestamp'], axis=1, inplace=True)
    return kmeans_cluster(df, k)
