import os
import numpy as np
from peak_extraction import *


def hash_func(vec, proj):
    bools = np.dot(proj, vec) > 0
    return bool2int(bools)

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        if j: y += 1<<i
    return y



class HashTable():

    def __init__(self, hash_size, dim):
        self.table = dict()
        self.hash_size = hash_size
        self.dim = dim
        self.projections = np.random.randn(self.hash_size, self.dim)

    def add(self, vec, label):
        entry = { "vector": None, "label": label }
        h = hash_func(vec, self.projections)
        if h in self.table:
            self.table[h].append(entry)
        else:
            self.table[h] = [entry]

    def query(self, vec):
        h = hash_func(vec, self.projections)
        if h in self.table:
            results = self.table[h]
        else:
            results = list()
        return results


class LSH():

    def __init__(self, dim):
        self.num_tables = 16
        self.hash_size = 16
        self.dim = dim
        self.tables = list()
        for i in range(self.num_tables):
            self.tables.append(HashTable(self.hash_size, self.dim))

    def add(self, vec, label):
        for table in self.tables:
            table.add(vec, label)

    def query(self, vec):
        results = list()
        for table in self.tables:
            results.extend(table.query(vec))
        return results


class FingerprintExtractor2():

    def __init__(self, source_dir, analysis_type="melspectrogram", sr=16000, n_fft=1024, hop_length=1024, peak_threshold=2.75):

        self.source_dir = source_dir
        self.analysis_type = analysis_type
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.peak_threshold = peak_threshold
        self.fv_size = 469
        self.lsh = LSH(self.fv_size)
        self.num_features_in_file = dict()
        for dirpath, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename.endswith(".wav"):
                    filepath = os.path.join(dirpath, filename)
                    self.num_features_in_file[filepath] = 0

    def store(self):
        analysis_type = self.analysis_type
        sr = self.sr
        n_fft = self.n_fft
        hop_length = self.hop_length
        threshold = self.peak_threshold

        for dirpath, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename.endswith(".wav"):
                    filepath = os.path.join(dirpath, filename)
                    audio, _ = librosa.load(filepath, sr=sr, mono=True)
                    spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                    for frame in spect:
                        self.lsh.add(frame, filepath)
                        self.num_features_in_file[filepath] += 1


    def query(self, filepath):
        analysis_type = self.analysis_type
        sr = self.sr
        n_fft = self.n_fft
        hop_length = self.hop_length
        threshold = self.peak_threshold

        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

        results = list()
        for frame in spect:
            results.extend(self.lsh.query(frame))

        counts = dict()
        for r in results:
            if r["label"] in counts:
                counts[r["label"]] += 1
            else:
                counts[r["label"]] = 1
        for k in counts:
            counts[k] = float(counts[k]) / self.num_features_in_file[k]
        return counts

