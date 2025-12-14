import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from cross_similarity import butter_bandpass_filter, plot_xsim_multi


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