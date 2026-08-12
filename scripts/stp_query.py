import os
import argparse
import numpy as np
import librosa
from joblib import Parallel, delayed
import spectromorphic_temporal_profile as stp
from data_loader import *
from taat import QueryResult, get_score_matrices, filter_result_for_path, get_merged_path, parse_seg_vals, SuppressRuntimeWarnings
from utils import yaml_read, json_write


parser = argparse.ArgumentParser(
    description="TAAT Spectromorphic Temporal Profile Query Script.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--project_dir", type=str, required=True, help="Path to the folder containing the audio files for analysis.", metavar="\b")
parser.add_argument("--config_file", type=str, default="scripts/stp.config.yaml", help="Path to the JSON config file.", metavar="\b")
parser.add_argument("--results_dir", type=str, default="./results", help="Directory in which to save exported JSON results files.", metavar="\b")
parser.add_argument("--cache_dir", type=str, default=None, help="Directory into which cached analysis data will be stored.", metavar="\b")
args = parser.parse_args()

config = yaml_read(args.config_file)
args = dict(list(vars(args).items()) + list(config.items()))

tape_speeds = {
    "simple": [12],
    "simple_variable": [4, 7, 12],
    "complex_variable": [2, 4, 5, 7, 9, 11, 12],
    "chromatic_variable": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
}

filepath1 = "/home/user/TAAT/data/chunks/001_End_of_the_World_(op.1)_chunk_2.wav"
filepath2 = "/home/user/TAAT/data/chunks/001_End_of_the_World_(op.1)_chunk_8.wav"
filepath3 = "/home/user/TAAT/data/Darude_Sandstorm/chunks/20_sec/Sandstorm_chunk_2.wav"
filepath4 = "/home/user/TAAT/data/Darude_Sandstorm/chunks/20_sec/Sandstorm_chunk_3.wav"
filepath5 = "/home/user/TAAT/data/under_pressure_chunks/Queen - Under Pressure_chunk_2.wav"
filepath6 = "/home/user/TAAT/data/under_pressure_chunks/Queen - Under Pressure_chunk_3.wav"
filepath7 = "/home/user/TAAT/data/ice_ice_baby_chunks/Vanilla Ice - Ice Ice Baby_chunk_1.wav"
filepath8 = "/home/user/TAAT/data/ice_ice_baby_chunks/Vanilla Ice - Ice Ice Baby_chunk_2.wav"
filepath9 = "/home/user/TAAT/data/ice_ice_baby_chunks/Vanilla Ice - Ice Ice Baby_chunk_3.wav"
filepath10 = "/home/user/TAAT/data/ice_ice_baby_chunks/Vanilla Ice - Ice Ice Baby_chunk_5.wav"
filepath11 = "/home/user/TAAT/data/Brahms_Hungarian_Dances/chunks/FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy_chunk_1.wav"
filepath12 = "/home/user/TAAT/data/Brahms_Hungarian_Dances/chunks/FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy_chunk_1_REVERSE.wav"
filepath13 = "/home/user/TAAT/data/Brahms_Hungarian_Dances/chunks/FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy_chunk_5.wav"
filepath14 = "/home/user/TAAT/data/chunks/001_End_of_the_World_(op.1)_chunk_5.wav"
filepath15 = "/home/user/TAAT/data/chunks/001_End_of_the_World_(op.1)_chunk_7.wav"
filepath16 = "/home/user/TAAT/data/Darude_Sandstorm/chunks/20_sec/Sandstorm_chunk_5.wav"
filepath17 = "/home/user/TAAT/data/mini_chunks/3 Daguerreo types/023 Daguerreo types, Op. 32B_chunk_2.wav"

def get_query_result(source_dir, query_filepath, sr=16000, chunk_length=30, overlap=0.5,
                     n_paths=1, enhance=True, zero_mean=False,
                     n_filters=5, no_identity_match=True, n_jobs=-1, cache_dir=None):
    #memory = Memory(cache_dir, verbose=0)
    #get_stp_match_v2 = memory.cache(stp.get_stp_match_v2)
    def check_identity_match(filename):
        return (no_identity_match and filename != os.path.basename(query_filepath) or (not no_identity_match))
    def print_fn(filename):
        return f"Computing spectromorphic cross-similarity for {os.path.basename(query_filepath)} against {os.path.basename(filename)}."
    matches = []
    jobs = []
    def stream_body(ref_filepath, ref_idx, ref_chunk, query_idx, query_chunk):
        (score, _, _, paths) = stp.get_stp_match_v2(query_chunk, ref_chunk,
                                                    sr=sr, n_paths=n_paths,
                                                    enhance=enhance, zero_mean=zero_mean, n_filters=n_filters)
        if score>0.0:
            paths, _ = stp.get_time_formatted_stp_paths(paths, sr=sr)
            query_segs = []
            ref_segs = []
            for (i, (ref_start, ref_stop, query_start, query_stop)) in enumerate(paths):
                query_segs.append([query_start, query_stop])
                ref_segs.append([ref_start, ref_stop])
            return {
                "score": score,
                "query_file": f"{os.path.basename(query_filepath)} chunk_{query_idx}",
                "query_segments": parse_seg_vals(query_segs),
                "collection_file": f"{ref_filepath} chunk_{ref_idx}",
                "collection_segments": parse_seg_vals(ref_segs)
            }
        else:
            return {
                "score": score,
                "query_file": f"{os.path.basename(query_filepath)} chunk_{query_idx}",
                "query_segments": [],
                "collection_file": f"{ref_filepath} chunk_{ref_idx}",
                "collection_segments": []
            }
    for (ref_filepath, ref_idx, ref_chunk) in walk(dir=source_dir, only_load_if=check_identity_match, print_fn=print_fn,
                                                   chunk_length=chunk_length, overlap=overlap, show_progress_bar=True):
        job = Parallel(n_jobs=n_jobs, return_as="generator")(delayed(stream_body)(ref_filepath, ref_idx, ref_chunk, query_idx, query_chunk) \
                  for (query_idx, query_chunk) in stream(query_filepath, chunk_length=chunk_length, overlap=overlap, show_progress_bar=False))
        jobs.append(job)
        for job in jobs:
            for match in job:
                matches.append(match)
    results = {}
    for k, entry in enumerate(matches):
        results[f"results_{k}"] = entry
    return results

def query(source_dir, query_filepath, sr=16000, chunk_length=10, overlap=0.5,
          n_paths=1, pitch_shift=0, prune=False, score_threshold=0.0, path_margin=2, no_identity_match=True, n_jobs=-1):
    info = {
        "source_dir": source_dir,
        "sr": sr,
        "chunk_length": chunk_length,
        "overlap": overlap,
        "n_paths": n_paths
    }
    qr = QueryResult(query_filepath=query_filepath,
                     result={},
                     info=info)
    n_query_chunks = get_n_chunks(query_filepath, chunk_length, overlap)
    if (n_query_chunks==1):
        warning_msg = f"Skipping analysis of query file '{query_filepath}' with chunk_length={chunk_length} "\
                        f"and pitch_shift={pitch_shift}, as there are too few chunks to process "\
                        "(must be more than a single chunk)."
        print(warning_msg)
        return qr
    with SuppressRuntimeWarnings():
        matches = get_query_result(source_dir=source_dir, query_filepath=query_filepath, sr=sr, chunk_length=chunk_length, overlap=overlap, n_paths=n_paths,
                                    enhance=False, zero_mean=prune, n_filters=5, no_identity_match=no_identity_match,
                                    n_jobs=n_jobs)
    mm = get_score_matrices(source_dir=source_dir,
                            query_filepath=query_filepath,
                            data=matches.values(),
                            chunk_length=chunk_length,
                            overlap=overlap,
                            threshold=score_threshold,
                            no_identity_match=no_identity_match)
    qr._result = matches
    qr.matrices = mm
    for i, (filepath, m) in enumerate(qr.matrices.items()):
        rqa, path = librosa.sequence.rqa(m)
        f = filter_result_for_path(filepath, qr._result.values(), path)
        merged = get_merged_path(f, chunk_length, overlap, path_margin)
        qr.result[f"results_{i}"] = merged
    return qr

def main():
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
                          overlap=args["overlap"], n_paths=args["n_paths"], prune=args["prune"],
                          score_threshold=0.0, path_margin=args["path_margin"], no_identity_match=args["no_identity_match"],
                          n_jobs=args["n_jobs"])
                #q = query(source_dir, query_filepath, sr=16000, chunk_length=10, overlap=0.5)
                result[f"tape_speed={tape_speed}, pitch_shift={key}"] = q.result
        output_filepath = os.path.join(results_dir, f"{case_name}-results.json")
        msg = f"Writing TAAT tape speed analysis data for {query_filepath} to {output_filepath}"
        json_write(result, output_filepath, msg)

if __name__ == "__main__":
    main()

