import json
import numpy as np
import argparse
from taat import query2


def list_of_str(arg):
    return arg.replace("'", "") \
        .replace('"', '') \
        .replace("[", "") \
        .replace("]", "") \
        .split(", ")[0] \
        .split(",")

parser = argparse.ArgumentParser(
    description="TAAT Pitch Shift Query Script.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--source_dir", type=str, required=True, help="Path to the folder containing the audio files for analysis.", metavar="\b")
parser.add_argument("--query_filepath", type=str, required=True, help="Query filepath.", metavar="\b")
parser.add_argument("--output_filepath", type=str, required=True, help="Path to the JSON file to write the results.", metavar="\b")
parser.add_argument("--sr", type=int, default=16000, help="Sample rate.", metavar="\b")
parser.add_argument("--features", type=list_of_str, default=["melspectrogram"], help="List of features to include in the analysis.", metavar="\b")
parser.add_argument("--chunk_length", type=int, default=10, help="Length (in seconds) of the audio chunk for analysis.", metavar="\b")
parser.add_argument("--overlap", type=float, default=0.5, help="Audio chunk overlap amount.", metavar="\b")
parser.add_argument("--n_fft", type=int, default=2048, help="FFT window size.", metavar="\b")
parser.add_argument("--hop_length", type=int, default=1024, help="FFT hop length.", metavar="\b")
parser.add_argument("--k", type=int, default=2, help="Number of nearest neighbours when computing the cross similarity.", metavar="\b")
parser.add_argument("--n_paths", type=int, default=5, help="Number of RQA paths.", metavar="\b")
parser.add_argument("--metric", type=str, default="cosine", help="Analysis metric.", metavar="\b")
parser.add_argument("--tape_speed", type=str, default="simple", help="Tape speed variations.", metavar="\b")
parser.add_argument("--prune", type=bool, default=False, help="Whether to prune off-diagonal regions in the RQA alignment matrix.", metavar="\b")
parser.add_argument("--score_threshold", type=float, default=0.25, help="Score threshold.", metavar="\b")
parser.add_argument("--path_margin", type=int, default=2, help="Tolerance margin (in seconds) between RQA paths.", metavar="\b")
parser.add_argument("--no_identity_match", type=bool, default=True, help="Whether or not to include the queried file in the result, if it is itself already present in the source folder.", metavar="\b")
args = parser.parse_args()

tape_speeds = {
    "simple": [12],
    "simple_variable": [4, 7, 12],
    "complex_variable": [2, 4, 5, 7, 9, 11, 12],
    "chromatic_variable": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
}

def main():
    tape_speed = args.tape_speed
    result = {}
    for ts in tape_speeds[tape_speed]:
        for s in [ts, -ts]:
            key = f"+{ts}" if s==ts else f"-{ts}"
            print(f"Running query for {tape_speed} tape speed, {key}.")
            q = query2(args.source_dir, args.query_filepath, sr=args.sr, chunk_length=args.chunk_length,
                       overlap=args.overlap, features=args.features, n_fft=args.n_fft, hop_length=args.hop_length,
                       k=args.k, metric=args.metric, n_paths=args.n_paths, pitch_shift=s, prune=args.prune,
                       score_threshold=args.score_threshold, path_margin=args.path_margin, no_identity_match=args.no_identity_match)
            result[f"tape_speed={tape_speed}, {key}"] = q.result
    with open(args.output_filepath, "w") as f:
        print(f"Writing TAAT tape speed analysis data for {args.query_filepath} to {args.output_filepath}")
        json.dump(result, f, indent=3)

if __name__ == "__main__":
    main()

'''
python scripts/tape_speed_query.py \
    --source_dir='data/full/temp2' \
    --query_filepath='data/full/temp2/0 slide-tape-no-1_original.wav' \
    --tape_speed='simple' \
    --output_filepath='tape_speed_query_test.json' \
    --features=["melspectrogram","spectral_centroid","chroma_cens"]
'''

