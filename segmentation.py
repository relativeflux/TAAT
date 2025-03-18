import numpy as np
import os
import librosa
from scipy.spatial import distance
from matplotlib import pyplot as plt

from audio import load
from features import extract_features

from ssm import compute_sm_from_filename, compute_novelty_ssm, plot_signal


# https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html

def cosine_distance(u, v):
    # Calculate cosine similarity
    cos_similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # Calculate cosine distance
    return 1 - cos_similarity

def distance_function(u, v):
    #return cosine_distance(u, v)
    return distance.cosine(u, v)

'''
def novelty_segmentation(data, sr, chunk_length):
    samples_per_chunk = sr * chunk_length
    r = range(0, len(data), samples_per_chunk)
    for i, j in zip(r, r[1:]):
        buffer0 = data[i:i+samples_per_chunk]
        buffer1 = data[j:j+samples_per_chunk]
    return True
'''

def novelty_segmentation(audio, sr, chunk_length, fft_size, hop_length):
    params = []
    samples_per_chunk = sr * chunk_length
    i = 0
    while i < len(audio):
        buffer = audio[i:i+samples_per_chunk]
        ext = extract_features("", i, buffer, sr, fft_size, hop_length)
        ext = [val for key, val in ext.items()
                   if key not in ['filename', 'timestamp']]
        params.append(ext)
        i += samples_per_chunk
    dim = len(params)
    m = np.zeros([dim, dim])
    r = range(0, dim)
    for i in r:
        u = params[i]
        for j in r:
            v = params[j]
            m[i,j] = distance_function(u, v)
    return m

def test(input):
    dim = len(input)
    m = np.zeros([dim, dim])
    r = range(0, dim)
    for i in r:
        val = input[i]
        for j in r:
            m[i,j] = val - input[j]
    return m

[3, 12, 1, 9]

def get_segmentation(filename, chunk_length, fft_size, hop_length):
    (_, sr, audio) = load(filename)
    seg = novelty_segmentation(audio, sr, chunk_length, fft_size, hop_length)
    plt.set_cmap("gray")
    plt.imshow(seg, interpolation="none")
    plt.show()

def get_ssm(filename, **kwargs):
    x, x_duration, X, Fs_X, S, I = compute_sm_from_filename(filename, **kwargs)
    plt.set_cmap("gray")
    plt.imshow(S, interpolation="none")
    plt.show()

########################################

def get_novelty_segmentation(filename, kernel=None, L_kernel=10, var=0.5, exclude=False, **kwargs):
    x, x_duration, X, Fs_X, S, I = compute_sm_from_filename(filename, **kwargs)
    seg = compute_novelty_ssm(S, kernel=kernel, L=L_kernel, var=var, exclude=exclude)
    return S, seg, Fs_X

'''
def plot_novelty_segmentation(S, seg, Fs_X):
    fig, ax, line = plot_signal(seg, Fs=Fs_X, color='k')
    plt.show()

def plot_novelty_segmentation2(S, seg, Fs_X):
    plt.set_cmap("gray")
    plt.imshow(S, interpolation="none")
    fig, ax, line = plot_signal(seg, Fs=Fs_X, color='k')
    plt.show()
'''

def plot_novelty_segmentation(save_to, S, seg, Fs_X):
    if not os.path.exists(save_to):
        os.makedirs(outdir)
    plt.set_cmap("gray")
    plt.imshow(S, interpolation="none")
    plt.savefig(os.path.join(save_to, "ssm.png"))
    fig, ax, line = plot_signal(seg, Fs=Fs_X, color='k')
    plt.savefig(os.path.join(save_to, "nov-seg.png"))
    

'''
file_path'../Dropbox/Miscellaneous/TAAT/Data/Test Cases/Test 1/data/001 End of the World (op.1).wav'
S, seg, Fs_X = get_novelty_segmentation(file_path, L_kernel=20, L=81, H=10, L_smooth=1, thresh=1)
plot_novelty_segmentation(S, seg, Fs_X)
'''

# file_path = 'scripts/FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.wav'