import os
import argparse
import tempfile
import numpy as np
import librosa
from taat import filter_silent_files, get_n_chunks
from data_loader import *
from utils import yaml_read

parser = argparse.ArgumentParser(
    description="TAAT dataset sanity check script.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--project_dir", type=str, required=True, help="Path to the folder containing the audio dataset to check.", metavar="\b")
parser.add_argument("--config_file", type=str, default="scripts/stp.config.yaml", help="Path to the JSON config file.", metavar="\b")
parser.add_argument("--silence_threshold", type=float, default=0.5, help="Silence threshold for the files in the dataset.", metavar="\b")
args = parser.parse_args()

config = yaml_read(args.config_file)
args = dict(list(vars(args).items()) + list(config.items()))

tape_speeds = {
    "None": [0],
    "simple": [12],
    "simple_variable": [4, 7, 12],
    "complex_variable": [2, 4, 5, 7, 9, 11, 12],
    "chromatic_variable": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
}

def dataset_sanity_check(source_dir, query_filepath, chunk_length=10, overlap=0.5, silence_threshold=0.5, tape_speed="simple"):
    res = filter_silent_files(source_dir, chunk_length, overlap=1.0,
                              filetype=".wav", threshold=silence_threshold)
    if res:
        print(f"The following files contain silence at or above the threshold of {silence_threshold}:")
        for filepath in res:
            print(filepath)
        print("Before running a TAAT query please remove the listed file(s) or, where appropriate, add amplification.")
    else:
        print(f"The Dataset at {source_dir} passed the sanity check, and it's safe to run a TAAT query using it.")
    '''
    if tape_speed:
        for ts in tape_speeds[tape_speed]:
            speeds = [ts] if tape_speed=="None" else [ts, -ts]
            for s in speeds:
                with tempfile.TemporaryDirectory() as tmpdir:
                    query_tmp_filepath = write_pitch_shifted_file(input_filepath=query_filepath,
                                                              output_dir=tmpdir,
                                                              pitch_shift=s)
                    n_query_chunks = get_n_chunks(query_tmp_filepath, chunk_length, overlap)
                    if (n_query_chunks==1):
                        print(f"Invalid tape speed {s} selected for query file {query_filepath}, as there would be too few chunks to process (must be more than a single chunk).")
    '''

def main():
    project_dir = args["project_dir"]
    project_dir = project_dir.replace("\\", "")
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
        dataset_sanity_check(source_dir=source_dir, query_filepath=query_filepath, chunk_length=args["chunk_length"], overlap=args["overlap"], silence_threshold=args["silence_threshold"])

if __name__ == "__main__":
    main()
