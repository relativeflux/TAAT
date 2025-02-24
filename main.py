import argparse
import os
import glob

from audio import stream #load_audio
from features import extract_features


DEFAULT_FEATURES = ["spectrogram",]
FFT_SIZE = 2048
HOP_LENGTH = 1024

def run(source_folder):
    features = []
    for fn in glob.glob(source_folder + '/*'):
        (_, sr, audio) = stream(fn)
        for buffer in audio:
            ext = extract_features(buffer, sr, FFT_SIZE, HOP_LENGTH)
            features.append(ext)
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tape Archive Analysis Toolkit (TAAT)")
    parser.add_argument("--source_folder", type=str, required=True, help="Path to the folder containing the audio files for analysis.")
    parser.add_argument("--features", nargs='+', type=str, default=DEFAULT_FEATURES, help="List of features to include in the analysis.")
    args = parser.parse_args()

    run(args.source_folder)