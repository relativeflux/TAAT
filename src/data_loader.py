import os
import math
import mmap
import numpy as np
import librosa
import wave
from tqdm import trange
from cross_similarity import get_xsim_multi2, get_time_formatted_paths, get_path_score
from taat import QueryResult, parse_query_output2, get_score_matrices, filter_result_for_path, get_merged_path


'''
def with_temp_file():
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write('stuff')
    finally:
        os.remove(path)
'''


def write_time_stretched_file(input_filepath: str, output_dir: str, sr=16000, chunk_length=10, time_stretch=1):
    if (not output_dir=="tmp"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    output_filepath = os.path.join(output_dir, os.path.basename(input_filepath))
    with wave.open(output_filepath, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        for (_, chunk) in stream(input_filepath, chunk_length=chunk_length, overlap=1.0,
                                 print_fn=False, show_progress_bar=False):
            if chunk is not None and len(chunk) != 0:
                chunk = librosa.effects.time_stretch(chunk, rate=time_stretch)
                chunk = (chunk * 32767).astype(np.int16)
                wav.writeframes(chunk)
            else:
                # If there's no more data, exit the loop
                print("Write operation completed")
                return None
    # This point should not be reached if the loop is exited correctly,
    # but if it does, it indicates the completion of writing
    print("Write operation completed")
    return None


def walk(dir="", filetype=".wav", only_load_if=lambda filename: filename==filename,
         chunk_length=30, overlap=0.5, show_progress_bar=True):
    result = []
    def print_fn(filepath):
        return f"File: {filepath}"
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(filetype):
                if callable(only_load_if)==True and only_load_if(filename)==True:
                    filepath = os.path.join(dirpath, filename)
                    for (idx, chunk) in stream(filepath, chunk_length=chunk_length, overlap=overlap, print_fn=print_fn,
                                               show_progress_bar=show_progress_bar):
                        yield((filepath, idx, chunk))

def stream(filepath, sr=16000, chunk_length=30, overlap=0.5, print_fn=None, show_progress_bar=True):
    overlap_secs = int(chunk_length * overlap)
    file_dur = int(librosa.get_duration(path=filepath))
    for (idx, offset) in enumerate(trange(0, math.floor(file_dur-overlap_secs), overlap_secs, disable=(not show_progress_bar))):
        if print_fn and idx==0:
            print(print_fn(filepath))
        chunk, _ = librosa.load(filepath, sr=sr, mono=True, offset=offset, duration=chunk_length)
        yield(idx, chunk)


def get_query_result(source_dir, query_filepath, sr=16000, chunk_length=30, overlap=0.5, features=["melspectrogram"],
                     n_fft=2048, hop_length=1024, k=5, metric="cosine", n_paths=5, time_stretch=1,
                     enhance=True, zero_mean=False, n_filters=5, no_identity_match=True):
    def check_identity_match(filename):
        return (no_identity_match and filename != os.path.basename(query_filepath) or (not no_identity_match))
    matches = {}
    for (ref_filepath, ref_idx, ref_chunk) in walk(dir=source_dir, only_load_if=check_identity_match,
                                                   chunk_length=chunk_length, overlap=overlap, show_progress_bar=False):
        (ref_xsim, ref_rqa, ref_paths, _) = get_xsim_multi2(ref_chunk, ref_chunk,
                                                            features=features, sr=sr,
                                                            fft_size=n_fft, hop_length=hop_length,
                                                            k=k, metric=metric, n_paths=n_paths,
                                                            enhance=enhance, zero_mean=zero_mean, n_filters=n_filters)
        for (query_idx, query_chunk) in stream(query_filepath, chunk_length=chunk_length, overlap=overlap, time_stretch=time_stretch):
            (query_xsim, query_rqa, query_paths, _) = get_xsim_multi2(query_chunk, ref_chunk,
                                                                      features=features, sr=sr,
                                                                      fft_size=n_fft, hop_length=hop_length,
                                                                      k=k, metric=metric, n_paths=n_paths,
                                                                      enhance=enhance, zero_mean=zero_mean, n_filters=n_filters)
            paths, _ = get_time_formatted_paths(query_paths, n_fft=n_fft, hop_length=hop_length)
            for (i, (ref_start, ref_stop, query_start, query_stop)) in enumerate(paths):
                match = {
                    "query_file": f"{os.path.basename(query_filepath)} chunk_{query_idx}",
                    "score": get_path_score(ref_rqa, query_rqa, ref_paths[i], query_paths[i]),
                    "queryStart": query_start,
                    "queryStop": query_stop,
                    "referenceStart": ref_start,
                    "referenceStop": ref_stop,
                }
                if f"{ref_filepath} chunk_{ref_idx}" not in matches:
                    matches[f"{ref_filepath} chunk_{ref_idx}"] = [match]
                else:
                    matches[f"{ref_filepath} chunk_{ref_idx}"].append(match)
    return matches

def query3(source_dir, query_filepath, sr=16000, chunk_length=30, overlap=0.5, features=["melspectrogram"],
           n_fft=2048, hop_length=1024, k=5, metric="cosine", n_paths=5, time_stretch=1,
           enhance=True, zero_mean=False, n_filters=5, no_identity_match=True):
    matches = get_query_result(source_dir=source_dir, query_filepath=query_filepath, sr=sr, chunk_length=chunk_length, overlap=overlap,
                               features=features, n_fft=n_fft, hop_length=hop_length, k=k, metric=metric, n_paths=n_paths,
                               time_stretch=time_stretch, enhance=enhance, zero_mean=zero_mean, n_filters=n_filters,
                               no_identity_match=no_identity_match)
    info = {
        "source_dir": source_dir,
        "features": features,
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "k": k,
        "metric": metric,
        "n_paths": n_paths
    }
    parsed_result = parse_query_output2(query_filepath, matches, n_paths)
    mm = get_score_matrices(source_dir=source_dir,
                            query_filepath=query_filepath,
                            data=parsed_result.values(),
                            chunk_length=chunk_length,
                            overlap=overlap,
                            no_identity_match=no_identity_match)
    qr = QueryResult(query_filepath=query_filepath,
                     result={},
                     info=info)
    qr._result = parsed_result
    qr.matrices = mm
    for i, (filepath, m) in enumerate(qr.matrices.items()):
        rqa, path = librosa.sequence.rqa(m)
        f = filter_result_for_path(filepath, qr._result.values(), path)
        merged = get_merged_path(f, chunk_length, overlap)
        qr.result[f"results_{i}"] = merged
    return qr
            




