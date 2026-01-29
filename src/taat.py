import os
from pathlib import Path
import pprint
import numpy as np
import sqlite3
import librosa
import soundfile as sf
from cross_similarity import *
from dtw import dtw


class QueryResult:
    """
    TAAT QueryResult class. Instances of this class are not intended to be created directly, instead they are created and returned by the TAAT _query_ function (see below).

    Methods
    -------

    export(output_filepath)
        Exports query result data to a file in JSON format.

    plot()
        Creates and displays a plot of the query result scores.

    pprint()
        Pretty prints the query result.

    write(outdir, format='wav')
        Writes query result data to disk as audio files.
    """

    def __init__(self, query_filepath, result, info):
        self.query_filepath = query_filepath
        self.info = info
        self.result = result

    def plot(self):
        """
        Creates and displays a plot of the query result scores.
        """
        result = self.result
        scores = {}
        ref_files = []
        for (_, val) in result.items():
            score = val["score"]
            ref_file = val["reference_file"]
            ref_files.append(ref_file)
            scores[ref_file] = score
        filename = os.path.basename(self.query_filepath)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[16,8])
        ax.plot(list(scores.values()), label="Path Cosine Scores")
        ax.legend()
        ax.set_xticks(range(0 ,len(ref_files)), labels=ref_files,
                      rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Scores")
        ax.set_title(f"Scores for query file '{filename}'")
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()

    def pprint(self):
        """
        Pretty prints the query result.
        """
        pprint.pp(self.result)

    def export(self, output_filepath):
        """
        Export TAAT query results to a JSON file.

        Parameters
        ----------

        **_output_filepath_ (str)**: Path to which to export the data. Should have the '.json' file extension.
        """
        with open(output_filepath, "w") as f:
            print(f"Writing TAAT analysis data for {self.query_filepath} to {output_filepath}")
            json.dump(self.result, f, indent=3)

    def write(self, outdir, format="wav"):
        """
        Write matches to disk as audio files.

        Parameters
        ----------

        **_outdir_ (str)**: Path to folder into which to write the audio data. The folder will be created if it doesn't exist.

        **_format_ (str), optional**: Audio file format.
        """
        query_filepath = self.query_filepath
        matches = self.result

        source_dir = self.info["source_dir"]
        features = self.info["features"]
        sr = self.info["sr"]
        n_fft = self.info["n_fft"]
        hop_length = self.info["hop_length"]
        k = self.info["k"]
        metric = self.info["metric"]
        n_paths = self.info["n_paths"]

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        info = dict(self.info)
        info["query_filepath"] = query_filepath
        with open(os.path.join(outdir, "info.json"), "w") as f:
            json.dump(info, f, indent=3)

        for match in matches.values():
            ref_filepath = os.path.join(source_dir, match["reference_file"])
            matches_dir = make_dir_for_file_path(outdir, ref_filepath)
            ref_segs = match["reference_segments"]
            query_segs = match["query_segments"]
            for (i, ref_seg) in enumerate(ref_segs):
                ref_start, ref_end = ref_seg
                query_start, query_end = query_segs[i]
                write_path_file(matches_dir, ref_filepath, f"ref_{i+1}", ref_start/1000, ref_end/1000, sr)
                write_path_file(matches_dir, query_filepath, f"query_{i+1}", query_start/1000, query_end/1000, sr)


def make_dir_for_file_path(parent_dir, file_path):
    file_path_dir = os.path.splitext(os.path.basename(file_path))[0]
    file_path_dir = ''.join(['_' if char == ' ' else char for char in file_path_dir])
    file_path_dir = os.path.join(parent_dir, file_path_dir)
    if not os.path.exists(file_path_dir):
        os.makedirs(file_path_dir)
    return file_path_dir

def run_backend(filepath1, filepath2, backend="cross_similarity", sr=16000, features=["melspectrogram"], n_fft=2048, hop_length=2048, metric="cosine", k=5, mode="affinity", n_paths=5, lowcut=180, highcut=3000, enhance=False, zero_mean=False, n_filters=10):
    sim_matrix = False
    rqa = False
    paths = []
    if os.path.basename(filepath1) != os.path.basename(filepath2):
        method = "cross similarity" if backend=="cross_similarity" else "DTW"
        print(f"Computing {method} for {os.path.basename(filepath1)} against {os.path.basename(filepath2)}.")
    if backend=="cross_similarity" or backend=="xsim":
        sim_matrix, rqa, paths, _ = get_xsim_multi(filepath1, filepath2, features=features, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, metric=metric, n_paths=n_paths, enhance=True, zero_mean=zero_mean, n_filters=n_filters)
    elif backend=="dtw":
        sim_matrix, rqa, paths = dtw(filepath1, filepath2, features=features, n_fft=n_fft, hop_length=hop_length, lowcut=lowcut, highcut=highcut, enhance=True)
    return sim_matrix, rqa, paths


def query(source_dir, query_filepath, backend="cross_similarity", features=["melspectrogram"], sr=16000, n_fft=2048, hop_length=1024, k=5, metric="cosine", n_paths=5, no_identity_match=True, verbose=False, zero_mean=False, n_filters=10):
    """
    Extracts feature data from the audio file supplied in _query_filepath_ and attempts to match it using cross-similarity scores with the audio files supplied in _source_dir_.

    Parameters
    ----------

    **_source_dir_ (str)**: Path to the folder of files whose feature data will be extracted.

    **_query_filepath_ (str)**: Path to the file to be queried against _source_dir_.

    **_backend_ (str)**: Set to one of 'cross_similarity' or 'dtw' (dynamic time warping).

    **_features_ (list[str]), optional**: List of features to extract in the analysis. Available features are: stft, melspectrogram, chroma_cqt, chroma_cens, mfcc, rms, tempogram, spectral_centroid, spectral_flatness, spectral_bandwidth and spectral_contrast.

    **_sr_ (int), optional**: Sample rate for the audio loaded for the analysis.

    **_n_fft_ (int), optional**: FFT analysis frame size.

    **_hop_length_ (int), optional**: FFT analysis hop length.

    **_k_ (int), optional**: Number of nearest-neighbours to compute for each analysis sample.

    **_metric_ (str), optional**: Distance metric to use for the cross-similarity analysis.

    **_n_paths_ (int), optional**: Number of RQA paths to compute.

    **_no_identity_match_ (bool), optional**: Whether or not to include the queried file in the result, if it is itself already present in the source folder.

    Returns
    -------

    **_query_result_ (QueryResult class instance)**
    """
    for dirpath, dirnames, filenames in os.walk(source_dir):
        matches = {}
        for filename in filenames:
            if filename.endswith(".wav"):
                if (no_identity_match and filename != os.path.basename(query_filepath)) or (not no_identity_match):
                    ref_filepath = os.path.join(dirpath, filename)
                    ref_xsim, ref_rqa, ref_paths = run_backend(ref_filepath, ref_filepath, backend=backend, features=features, sr=sr, n_fft=n_fft, hop_length=hop_length, k=k, metric=metric, n_paths=n_paths, enhance=True, zero_mean=zero_mean, n_filters=n_filters)
                    query_xsim, query_rqa, query_paths = run_backend(query_filepath, ref_filepath, backend=backend, features=features, sr=sr, n_fft=n_fft, hop_length=hop_length, k=k, metric=metric, n_paths=n_paths, enhance=True, zero_mean=zero_mean, n_filters=n_filters)
                    paths, _ = get_time_formatted_paths(query_paths, n_fft=n_fft, hop_length=hop_length)
                    for (i, (ref_start, ref_stop, query_start, query_stop)) in enumerate(paths):
                        match = {
                            "score": get_path_score(ref_rqa, query_rqa, ref_paths[i], query_paths[i]),
                            "queryStart": query_start,
                            "queryStop": query_stop,
                            "referenceStart": ref_start,
                            "referenceStop": ref_stop,
                        }
                        if ref_filepath not in matches:
                            matches[ref_filepath] = [match]
                        else:
                            matches[ref_filepath].append(match)
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
        if verbose:
            return QueryResult(query_filepath=query_filepath,
                               result=matches,
                               info=info)
        else:
            parsed_result = parse_query_output(query_filepath, matches)
            return QueryResult(query_filepath=query_filepath,
                               result=parsed_result,
                               info=info)

def query2(source_dir, query_filepath, chunk_length=30, overlap=0.5,
           features=["melspectrogram"], sr=16000, n_fft=2048, hop_length=1024,
           k=5, metric="cosine", n_paths=5, no_identity_match=True, verbose=False,
           zero_mean=False, n_filters=10):
    for dirpath, dirnames, filenames in os.walk(source_dir):
        query_file_dur = int(librosa.get_duration(path=query_filepath))
        matches = {}
        for filename in filenames:
            if filename.endswith(".wav"):
                if (no_identity_match and filename != os.path.basename(query_filepath)) or (not no_identity_match):
                    ref_filepath = os.path.join(dirpath, filename)
                    ref_file_dur = int(librosa.get_duration(path=ref_filepath))
                    for (ref_idx, ref_offset) in enumerate(range(0, math.floor(ref_file_dur-chunk_length), int(chunk_length * overlap))):
                        ref_chunk, _ = librosa.load(ref_filepath, sr=sr, mono=True, offset=ref_offset, duration=chunk_length)
                        ref_xsim, ref_rqa, ref_paths, _ = get_xsim_multi(ref_chunk, ref_chunk, features=features, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, metric=metric, n_paths=n_paths, enhance=True, zero_mean=zero_mean, n_filters=n_filters)
                        for (query_idx, query_offset) in enumerate(range(0, math.floor(query_file_dur-chunk_length), int(chunk_length * overlap))):
                            query_chunk, _ = librosa.load(query_filepath, sr=sr, mono=True, offset=query_offset, duration=chunk_length)
                            print(f"Computing cross-similarity for {os.path.basename(query_filepath)} chunk {query_idx} against {os.path.basename(ref_filepath)} chunk_{ref_idx}.")
                            query_xsim, query_rqa, query_paths, _ = get_xsim_multi(query_chunk, ref_chunk, features=features, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, metric=metric, n_paths=n_paths, enhance=True, zero_mean=zero_mean, n_filters=n_filters)
                            paths, _ = get_time_formatted_paths(query_paths, n_fft=n_fft, hop_length=hop_length)
                            for (i, (ref_start, ref_stop, query_start, query_stop)) in enumerate(paths):
                                match = {
                                    "query_file": os.path.basename(query_filepath),
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
        if verbose:
            return QueryResult(query_filepath=query_filepath,
                               result=matches,
                               info=info)
        else:
            parsed_result = parse_query_output2(query_filepath, matches, n_paths)
            return QueryResult(query_filepath=query_filepath,
                               result=parsed_result,
                               info=info)

def parse_query_output2(query_filename, query_output, n_paths):
    temp = []
    for i, (key, entry) in enumerate(query_output.items()):
        for j in range(0, int(len(entry)/n_paths)):
            val = f"{query_filename}_chunk_{j}"
            filtered = list(filter(lambda x: "query_file" in x and x["query_file"]==val, entry))
            score = float(np.mean([d["score"] for d in filtered]))
            query_segs = [[d["queryStart"]*1000, d["queryStop"]*1000] for d in filtered]
            ref_segs = [[d["referenceStart"]*1000, d["referenceStop"]*1000] for d in filtered]
            temp.append({
                "score": score,
                "query_file": val,
                "query_segments": query_segs,
                "reference_file": key,
                "reference_segments": ref_segs
            })
    results = {}
    for k, entry in enumerate(temp):
        results[f"results_{k}"] = entry
    return results

def get_n_chunks(filepath, chunk_length, overlap):
    dur = librosa.get_duration(path=filepath)
    return dur // chunk_length // overlap

