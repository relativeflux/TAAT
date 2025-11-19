import os
from pathlib import Path
import numpy as np
import sqlite3
import copy
import scipy.signal as signal
import librosa
import soundfile as sf
from cross_similarity import *


class TAAT:

    def __init__(self, source_dir, features=["melspectrogram"], sr=16000, n_fft=2048, hop_length=1024, k=5, metric="cosine", num_paths=5):

        self.source_dir = source_dir
        self.features = features
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.k = k
        self.metric = metric
        self.num_paths = num_paths
        self.data = {}


    def store(self):
        features = self.features
        sr = self.sr
        n_fft = self.n_fft
        hop_length = self.hop_length
        k = self.k
        metric = self.metric
        num_paths = self.num_paths

        for dirpath, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename.endswith(".wav"):
                    filepath = os.path.join(dirpath, filename)
                    print(f"Computing cross-similarity for {filename} against itself.")
                    xsim, rqa, paths, _ = get_xsim_multi(filepath, filepath, features=features, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, metric=metric, num_paths=num_paths, enhance=True)
                    self.data[filename] = [xsim, rqa, paths] #get_path_data(rqa, paths[0])
    

    def query(self, query_filepath, no_identity_match=True, verbose=False):
        features = self.features
        sr = self.sr
        n_fft = self.n_fft
        hop_length = self.hop_length
        k = self.k
        metric = self.metric
        num_paths = self.num_paths
        matches = {}

        for ref_filename in self.data:
            ref_filepath = os.path.join(self.source_dir, ref_filename)
            if no_identity_match==True and ref_filename != os.path.basename(query_filepath):
                ref_xsim, ref_rqa, ref_paths = self.data[ref_filename]
                print(f"Computing cross-similarity for {os.path.basename(query_filepath)} against {ref_filename}.")
                query_xsim, query_rqa, query_paths, _ = get_xsim_multi(ref_filepath, query_filepath, features=features, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, metric=metric, num_paths=num_paths, enhance=True)
                paths, _ = get_time_formatted_paths(query_paths, n_fft=n_fft, hop_length=hop_length)
                for (i, (ref_start, ref_stop, query_start, query_stop)) in enumerate(paths):
                    match = {
                        "score": {
                            "path_score": get_path_score(ref_rqa, query_rqa, ref_paths[i], query_paths[i]),
                            "norm_score": get_normalized_score(query_rqa, query_paths[0])
                        },
                        "queryStart": query_start,
                        "queryStop": query_stop,
                        "referenceStart": ref_start,
                        "referenceStop": ref_stop,
                    }
                    if ref_filepath not in matches:
                        matches[ref_filepath] = [match]
                    else:
                        matches[ref_filepath].append(match)
        if verbose:
            return matches
        else:
            return self.parse_query_output(query_filepath, matches)

    def parse_query_output(self, query_filepath, query_output):
        result = {}
        keys = list(query_output.keys())
        for (i, v) in enumerate(list(query_output.values())):
            k = keys[i]
            #score = float(np.mean([match["score"] for match in v]))
            scores = [match["score"] for match in v]
            query_segs = [[match["queryStart"]*1000, match["queryStop"]*1000] for match in v]
            ref_segs = [[match["referenceStart"]*1000, match["referenceStop"]*1000] for match in v]
            result[f"results_{i}"] = {
                "score": scores[0], #score,
                "query_file": os.path.basename(query_filepath),
                "query_segments": query_segs,
                "reference_file": os.path.basename(k),
                "reference_segments": ref_segs
            }
        return result