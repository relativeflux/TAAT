import os
import sqlite3
from peak_extraction import find_peaks


class Matcher:

    def __init__(self, dp_path):

    
    def match(self, query_path):
        peaks = find_peaks(query_path, sr, n_fft, hop_length, threshold)
        self.peaks = [
            {'time': t, 'bin': f, 'magnitude': m, 'usages': 0} for (t, f, m) in peaks
        ]
        self.query_path = query_path
        self.extract_fingerprints() # ????