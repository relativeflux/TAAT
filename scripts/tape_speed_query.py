import os
import json
import numpy as np
import argparse
from utils import yaml_read, json_write
from taat import query


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
parser.add_argument("--project_dir", type=str, required=True, help="Path to the folder containing the audio files for analysis.", metavar="\b")
parser.add_argument("--config_file", type=str, default="scripts/default.config.yaml", help="Path to the JSON config file.", metavar="\b")
parser.add_argument("--results_dir", type=str, default="./results", help="Directory in which to save exported JSON results files.", metavar="\b")
parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory into which cached analysis data will be stored.", metavar="\b")
args = parser.parse_args()

#config = json_read(args.config_file)
config = yaml_read(args.config_file)
args = dict(list(vars(args).items()) + list(config.items()))

tape_speeds = {
    "simple": [12],
    "simple_variable": [4, 7, 12],
    "complex_variable": [2, 4, 5, 7, 9, 11, 12],
    "chromatic_variable": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
}

def main(args):
    project_dir = args["project_dir"]
    project_dir = project_dir.replace("\\", "")
    results_dir = args["results_dir"]
    results_dir = results_dir.replace("\\", "")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for case_name in sorted(os.listdir(project_dir)):
        case_path = os.path.join(project_dir, case_name)
        if not os.path.isdir(case_path):
            continue
        source_dir = os.path.join(case_path, "data")
        source_dir = source_dir.replace("\\", "")
        input_dir = os.path.join(case_path, "input")
        input_dir = input_dir.replace("\\", "")
        if not os.path.isdir(source_dir) or not os.path.isdir(input_dir):
            print(f"Skipping {case_name}: missing data or input directory")
            continue
        # Assume exactly one input file
        input_files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        if len(input_files) != 1:
            print(f"Skipping {case_name}: expected 1 input file, found {len(input_files)}")
            continue
        query_filepath = os.path.join(input_dir, input_files[0])
        tape_speed = args["tape_speed"]
        result = {}
        for ts in tape_speeds[tape_speed]:
            for s in [0, ts, -ts]:
                key = 0
                if s!=0:
                    if s==ts:
                        key = f"+{ts}"
                    else:
                        key = f"-{ts}"
                print(f"Running query for {tape_speed} tape speed, pitch_shift={key}.")
                q = query(source_dir, query_filepath, sr=args["sr"], chunk_length=args["chunk_length"],
                          overlap=args["overlap"], features=args["features"], n_fft=args["n_fft"], hop_length=args["hop_length"],
                          k=args["k"], metric=args["metric"], n_paths=args["n_paths"], pitch_shift=s, prune=args["prune"],
                          score_threshold=args["score_threshold"], path_margin=args["path_margin"], no_identity_match=args["no_identity_match"],
                          n_jobs=args["n_jobs"], cache_dir=args["cache_dir"])
                result[f"tape_speed={tape_speed}, pitch_shift={key}"] = q.result
        output_filepath = os.path.join(results_dir, f"{case_name}-results.json")
        msg = f"Writing TAAT tape speed analysis data for {query_filepath} to {output_filepath}"
        json_write(result, output_filepath, msg)

if __name__ == "__main__":
    main(args)

'''
python scripts/tape_speed_query.py \
    --project_dir="data/full/temp3" \
    --config_file=./custom.config.json
'''

