import argparse
import os
import glob
import json

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from audio import load, stream #load_audio
from features import extract_features
from clustering import get_clusters


OUTPUT_DEST = None #"stdout"
DEFAULT_FEATURES = ["spectrogram",]
CHUNK_LENGTH = 8
FFT_SIZE = 2048
HOP_LENGTH = 512


def save_outputs(dir, features, labels):
    # Create output directory.
    outdir = os.path.join("./output", dir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Save feature data in JSON format.
    with open(os.path.join(outdir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=3)
    with open(os.path.join(outdir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=3)

def show_dendrogram(linkage_matrix, title="", xlabel="", **kwargs):
    plt.title(title)
    dendrogram(linkage_matrix, truncate_mode="level", p=3)
    plt.xlabel(xlabel)
    plt.show()

def parse_features(features):
    parsed = []
    for (_, value) in features.items():
        for dict in value:
            lis = [val for _, val in dict.items()]
            parsed.append(lis)
    return parsed

def sort_source_files(source_folder):
    l = glob.glob(source_folder + '/*')
    return sorted(l, key=lambda f: int(f.partition("(op.")[2].partition(")")[0]))

def run(source_folder,
        output_dest=OUTPUT_DEST,
        chunk_length=CHUNK_LENGTH,
        fft_size=FFT_SIZE,
        hop_length=HOP_LENGTH,
        *args, **kwargs):
    features = {} #[]
    #for filename in glob.glob(source_folder + '/*'):
    for filename in sort_source_files(source_folder):
        features[filename] = []
        (_, sr, audio) = load(filename)
        samples_per_chunk = sr * chunk_length
        i = 0
        print(f"Extracting features for file {filename}")
        while i < len(audio):
            buffer = audio[i:i+samples_per_chunk]
            ext = extract_features(buffer, sr, fft_size, hop_length)
            #ext = [value.item() for _, value in ext.items()]
            features[filename].append(ext)
            i += samples_per_chunk
    X = parse_features(features)
    print("Computing clusters...")
    (labels, linkage_matrix) = get_clusters(X)
    labels = labels.tolist()
    output_folder_name = os.path.basename(source_folder)
    print("Saving features to disk...")
    save_outputs(output_folder_name, features, labels)
    plt.title(output_folder_name)
    dendrogram(linkage_matrix, truncate_mode="level", p=3)
    #plt.xlabel(xlabel)
    plt.savefig(os.path.join("./output", output_folder_name, "clusters.png"))
    print("Done.")
    '''
    if output_dest == None:
        save_outputs(source_folder, features)
    elif output_dest == "stdout"
        return features
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tape Archive Analysis Toolkit (TAAT)")
    parser.add_argument("--source_folder", type=str, required=True, help="Path to the folder containing the audio files for analysis.")
    parser.add_argument("--output_dest", type=str, default=OUTPUT_DEST, help="Output destination. If 'stdout', output is printed to standard output; if None, output is saved to the provided TAAT 'output' directory; if a directory path, output is saved to that destination.")
    parser.add_argument("--features", nargs='+', type=str, default=DEFAULT_FEATURES, help="List of features to include in the analysis.")
    parser.add_argument("--chunk_length", type=int, default=CHUNK_LENGTH, help="Length (in seconds) of the audio chunk for analysis.")
    parser.add_argument("--fft_size", type=int, default=FFT_SIZE, help="FFT size.")
    parser.add_argument("--hop_length", type=int, default=HOP_LENGTH, help="FFT hop length.")
    args = parser.parse_args()

    run(args.source_folder, args.output_dest, args.chunk_length, args.fft_size, args.hop_length)