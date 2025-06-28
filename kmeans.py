import os
import sklearn
import numpy as np
import pandas as pd
import librosa
import matplotlib.cm as cm
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
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
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
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
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

def silhouette_analysis(df, r=[2, 6]):
    for k in range(r[0], r[1]):
        # Create a subplot with 1 row and 2 columns.
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        # The 1st subplot is the silhouette plot.
        ax1.set_xlim([-0.1, 1])
        # The (k+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(df) + (k + 1) * 10])

        # Run the clustering
        kmeans = sklearn.cluster.KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(df)

        # Run the silhouette analysis...
        silhouette_avg = sklearn.metrics.silhouette_score(df, clusters)
        print(
            "For k =",
            k,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        sample_silhouette_values = sklearn.metrics.silhouette_samples(df, clusters)

        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([]) # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(clusters.astype(float) / k)
        ax2.scatter(
            df.iloc[:, 0], df.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with k = %d"
            % k,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()
