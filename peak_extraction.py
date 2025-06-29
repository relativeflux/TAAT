import numpy as np
import statistics
import scipy
import librosa
import matplotlib.pyplot as plt


'''
Adapted from https://williamsantos.me/posts/2023/spectrogram-peak-detection-with-scipy/

peak = (time, frequency, intensity)

The steps are as follows:

1. Filter out zero values.
2. Calculate mean and standard deviation of intensities.
3. Assign a z score to each intensity in the spectrogram matrix.
4. Create a label matrix from the frequency intensities that are above threshold parameter.
5. For each isolated region in the label matrix, extract the max intensity value from the original spectrogram.
6. Create a list of peak values and their location.
'''
def find_peaks(filepath, sr=16000, n_fft=2048, hop_length=1024, threshold=2.75):
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    db_mel_spect = librosa.amplitude_to_db(mel_spect, ref=np.max)
    # Remove zero values.
    flattened = np.matrix.flatten(db_mel_spect)
    filtered = flattened[flattened > np.min(flattened)]
    # Create a normal distribution from frequency intensities
    # then map a zscore onto each intensity value.
    ndist = statistics.NormalDist(np.mean(filtered), np.std(filtered))
    zscore = np.vectorize(lambda x: ndist.zscore(x))
    zscore_matrix = zscore(db_mel_spect)
    # Create label matrix from frequency intensities that are
    # above threshold.
    mask_matrix = zscore_matrix > threshold
    labelled_matrix, num_regions = scipy.ndimage.label(mask_matrix)
    label_indices = np.arange(num_regions) + 1
    # For each isolated region in the mask, identify the maximum
    # value, then extract its position.
    peak_positions = scipy.ndimage.maximum_position(
        zscore_matrix, labelled_matrix, label_indices)
    # Create list of peaks (time, frequency, intensity).
    peaks = [(x, y, db_mel_spect[y, x]) for y, x in peak_positions]
    return db_mel_spect, peaks

def plot_peaks(spect, peaks, figsize=[16,8], s=1.5, color="red"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        x=[p[0] for p in peaks],
        y=[p[1] for p in peaks],
        s=s,
        color=color
    )
    ax.matshow(spect[:500, :600])
    plt.show()
