import numpy as np
import librosa
from scipy.spatial import distance
from matplotlib import pyplot as plt

from audio import load
from features import extract_features

from ssm import compute_sm_from_filename


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
    x, x_duration, X, Fs_feature, S, I = compute_sm_from_filename(filename, **kwargs)
    plt.set_cmap("gray")
    plt.imshow(S, interpolation="none")
    plt.show()