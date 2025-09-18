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
def find_peaks(filepath, analysis_type="stft", sr=16000, n_fft=2048, hop_length=1024, threshold=2.75):
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    spect = False
    if analysis_type=="stft":
        spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spect = np.abs(spect)
    elif analysis_type=="melspectrogram":
        spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    else:
        spect = librosa.cqt(audio, sr=sr, hop_length=hop_length)
    spect = librosa.amplitude_to_db(spect, ref=np.max)
    # Remove zero values.
    flattened = np.matrix.flatten(spect)
    filtered = flattened[flattened > np.min(flattened)]
    # Create a normal distribution from frequency intensities
    # then map a zscore onto each intensity value.
    ndist = statistics.NormalDist(np.mean(filtered), np.std(filtered))
    zscore = np.vectorize(lambda x: ndist.zscore(x))
    zscore_matrix = zscore(spect)
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
    peaks = [(int(x), int(y), float(spect[y, x])) for y, x in peak_positions]
    return spect, peaks

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
    peaks = [(int(x), int(y), float(db_mel_spect[y, x])) for y, x in peak_positions]
    return db_mel_spect, peaks
'''

def plot_peaks(spect, peaks, peaks_only=False, figsize=[16,8], s=1.5, color="red"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        x=[p[0] for p in peaks],
        y=[p[1] for p in peaks],
        s=s,
        color=color
    )
    bg = np.zeros(spect.shape) if peaks_only else spect
    ax.matshow(bg[:500, :600])
    plt.show()

def parse_peaks(peaks, n_fft=2048, hop_length=1024):
    peaks = np.array(sorted(peaks))
    num_samples = librosa.frames_to_samples(peaks[-1][0], n_fft=n_fft, hop_length=hop_length)
    result = np.zeros(num_samples+1)
    frames = librosa.frames_to_samples(peaks[:,0], n_fft=n_fft, hop_length=hop_length)
    amps = peaks[:,2]
    for (i, f) in enumerate(frames):
        result[f] = amps[i]
    return result


def get_peaks_xsim(comp_path, ref_path, analysis_type="melspectrogram", sr=16000, n_fft=2048, hop_length=1024, peak_threshold=2.75, k=2, metric='euclidean', mode='affinity'):
    ref_spect, ref_peaks = find_peaks(ref_path, analysis_type, sr, n_fft, hop_length, peak_threshold)
    ref_zeros = np.zeros(ref_spect.shape)
    for [t, f, m] in ref_peaks:
        ref_zeros[f][t] = m
    comp_spect, comp_peaks = find_peaks(comp_path, analysis_type, sr, n_fft, hop_length, peak_threshold)
    comp_zeros = np.zeros(comp_spect.shape)
    for [t, f, m] in comp_peaks:
        comp_zeros[f][t] = m
    x_ref = librosa.feature.stack_memory(ref_zeros, n_steps=10, delay=3)
    x_comp = librosa.feature.stack_memory(comp_zeros, n_steps=10, delay=3)
    return librosa.segment.cross_similarity(x_comp, x_ref, k=k, metric=metric, mode=mode)

'''
from cross_similarity import plot_cross_similarity_matrix

xsim = get_peaks_xsim('test5/16000kHz/chunks/001_End_of_the_World_(op.1)_chunk_7.wav', 'test5/16000kHz/chunks/005_Disintegration_(op.10)_chunk_9.wav', peak_threshold=0.1, metric="cosine")

plot_cross_similarity_matrix(xsim, hop_length=1024)
'''

import os
import soundfile as sf

def get_layer(filepath, outdir, analysis_type="stft", sr=16000, n_fft=2048, hop_length=1024, peak_threshold=2.75):
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    spect = False
    if analysis_type=="stft":
        spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    else:
        spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    _, peaks = find_peaks(filepath, analysis_type=analysis_type, sr=sr, n_fft=n_fft, hop_length=hop_length, threshold=peak_threshold)
    phase_stft = np.angle(spect)
    zeros = np.zeros(spect.shape)
    for [t, f, m] in peaks:
        zeros[f][t] = m
    reconstruct = zeros * np.exp(1j * phase_stft)
    layer = librosa.istft(reconstruct, n_fft=n_fft, hop_length=hop_length)
    sf.write(os.path.join(outdir, os.path.basename(filepath)), layer, sr)