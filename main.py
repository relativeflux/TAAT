import argparse
import os
import glob

from audio import load, stream #load_audio
from features import extract_features


DEFAULT_FEATURES = ["spectrogram",]
FFT_SIZE = 2048
HOP_LENGTH = 512
CHUNK_LENGTH = 8


def run(source_folder, chunk_length, fft_size, hop_length):
    features = []
    for filename in glob.glob(source_folder + '/*'):
        (_, sr, audio) = load(filename)
        samples_per_chunk = sr * chunk_length
        i = 0
        while i < len(audio):
            buffer = audio[i:i+samples_per_chunk]
            ext = extract_features(buffer, sr, fft_size, hop_length)
            ext = [value.item() for _, value in ext.items()]
            features.append(ext)
            i += samples_per_chunk
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tape Archive Analysis Toolkit (TAAT)")
    parser.add_argument("--source_folder", type=str, required=True, help="Path to the folder containing the audio files for analysis.")
    parser.add_argument("--features", nargs='+', type=str, default=DEFAULT_FEATURES, help="List of features to include in the analysis.")
    parser.add_argument("--chunk_length", type=int, default=CHUNK_LENGTH, help="Length (in seconds) of the audio chunk for analysis.")
    parser.add_argument("--fft_size", type=int, default=FFT_SIZE, help="FFT size.")
    parser.add_argument("--hop_length", type=int, default=HOP_LENGTH, help="FFT hop length.")
    args = parser.parse_args()

    run(args.source_folder, args.chunk_length, args.fft_size, args.hop_length)