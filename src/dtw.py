import os
import numpy as np
import librosa
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from cross_similarity import butter_bandpass_filter, apply_features, plot_xsim_multi


def dtw(filepath1, filepath2, sr=16000, features=["melspectrogram"], n_fft=2048, hop_length=1024, lowcut=180, highcut=3000, enhance=True):
    samples1, _ = librosa.load(filepath1, sr=sr, mono=True)
    samples2, _ = librosa.load(filepath2, sr=sr, mono=True)
    samples1 = butter_bandpass_filter(samples1, lowcut=lowcut, highcut=highcut, sr=sr)
    samples2 = butter_bandpass_filter(samples2, lowcut=lowcut, highcut=highcut, sr=sr)
    X = apply_features(samples1, features=features, sr=sr, n_fft=n_fft, hop_length=hop_length)
    Y = apply_features(samples2, features=features, sr=sr, n_fft=n_fft, hop_length=hop_length)
    dist_matrix, path = librosa.sequence.dtw(X, Y, subseq=True)
    dist_matrix = dist_matrix / np.max(dist_matrix)
    sim_matrix = 1 - dist_matrix
    if enhance:
        sim_matrix = librosa.segment.path_enhance(sim_matrix, 64, n_filters=10)
    rqa, path = librosa.sequence.rqa(sim_matrix, gap_onset=5, gap_extend=25)
    return sim_matrix, rqa, [path]

'''
def plot_dtw(D, wp, filepath1, filepath2, hop_length=1024):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames', ax=ax[0], hop_length=hop_length)
    ax[0].set(title='DTW cost matrix', xlabel=f"{os.path.basename(filepath1)}", ylabel=f"{os.path.basename(filepath2)}")
    ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='yellow')
    ax[0].legend()
    fig.colorbar(img, ax=ax[0])
    ax[1].plot(D[-1, :] / wp.shape[0])
    #ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2], title='Matching cost function')
    plt.show()
'''

def plot_dtw(sim_matrix, rqa, paths):
    plot_xsim_multi(sim_matrix, rqa, paths)

####################################################################################

def get_dtw_match(filepath1, filepath2, sr=16000, hop_length=512):
    # Load audio files
    y1, _ = librosa.load(filepath1, sr=sr, mono=True)
    y2, _ = librosa.load(filepath2, sr=sr, mono=True)
    # Extract Robust Features (Chroma STFT focuses on pitch/harmony)
    # Transpose them so that frames are along the rows (axis 0) for distance calculations
    chroma1 = librosa.feature.chroma_stft(y=y1, sr=sr, hop_length=hop_length).T
    chroma2 = librosa.feature.chroma_stft(y=y2, sr=sr, hop_length=hop_length).T
    # Compute Cross-Distance Cost Matrix
    # Using "cosine" distance makes the comparison invariant to volume/amplitude differences
    cost_matrix = cdist(chroma1, chroma2, metric="cosine")
    # Perform Dynamic Time Warping
    # librosa.sequence.dtw finds the minimum cost path through the cost matrix
    # subseq=True allows for partial matching if one file is a subset of the other
    D, path = librosa.sequence.dtw(C=cost_matrix, subseq=False)
    # Extract alignment coordinates
    # The path returned by librosa is inverted (starts from the end), so we flip it
    path_flipped = np.flip(path, axis=0)
    path_x_coords = path_flipped[:, 0]
    path_y_coords = path_flipped[:, 1]
    # 6. Calculate a Normalized Similarity Score
    # Total minimum alignment cost
    total_cost = D[-1, -1] 
    # Worst-case scenario cost (if every step suffered maximum cosine distance of 2.0)
    max_possible_cost = 2.0 * len(path_x_coords)
    # Convert distance into a normalized similarity percentage (0.0 to 1.0)
    normalized_score = 1.0 - (total_cost / max_possible_cost)
    # Ensure precision bounds
    normalized_score = float(np.clip(normalized_score, 0.0, 1.0))
    return normalized_score, path_x_coords, path_y_coords

