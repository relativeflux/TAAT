import os
from pathlib import Path
import pprint
import numpy as np
import sqlite3
import librosa
import soundfile as sf
from cross_similarity import *


class QueryResult:
    """
    TAAT QueryResult class. Instances of this class are not intended to be created directly, instead they are created and returned by the TAAT _query_ function (see below).

    Methods
    -------

    export(output_filepath)
        Exports query result data to a file in JSON format.

    plot()
        Stores extracted cross-similarity data from a folder of audio files in the database.

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
        num_paths = self.info["num_paths"]

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

def query(source_dir, query_filepath, features=["melspectrogram"], sr=16000, n_fft=2048, hop_length=1024, k=5, metric="cosine", num_paths=5, no_identity_match=True, verbose=False):
    """
    Extracts feature data from the audio file supplied in _query_filepath_ and attempts to match it using cross-similarity scores with the audio files supplied in _source_dir_.

    Parameters
    ----------

    **_source_dir_ (str)**: Path to the folder of files whose feature data will be extracted.

    **_query_filepath_ (str)**: Path to the file to be queried against _source_dir_.

    **_features_ (list[str]), optional**: List of features to extract in the analysis. Available features are: stft, melspectrogram, chroma_cqt, mfcc, rms, tempogram, spectral_centroid, spectral_flatness, spectral_bandwidth and spectral_contrast.

    **_sr_ (int), optional**: Sample rate for the audio loaded for the analysis.

    **_n_fft_ (int), optional**: FFT analysis frame size.

    **_hop_length_ (int), optional**: FFT analysis hop length.

    **_k_ (int), optional**: Number of nearest-neighbours to compute for each analysis sample.

    **_metric_ (str), optional**: Distance metric to use for the cross-similarity analysis.

    **_num_paths_ (int), optional**: Number of RQA paths to compute.

    **_no_identity_match_ (bool), optional**: Whether or not to include the queried file in the result, if it is itself already stored in the database.

    Returns
    -------

    **_query_result_ (QueryResult class instance)**
    """
    for dirpath, dirnames, filenames in os.walk(source_dir):
        matches = {}
        for filename in filenames:
            if filename.endswith(".wav"):
                if no_identity_match==True and filename != os.path.basename(query_filepath):
                    ref_filepath = os.path.join(dirpath, filename)
                    ref_xsim, ref_rqa, ref_paths, _ = get_xsim_multi(ref_filepath, ref_filepath, features=features, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, metric=metric, num_paths=num_paths, enhance=True)
                    print(f"Computing cross-similarity for {os.path.basename(query_filepath)} against {filename}.")
                    query_xsim, query_rqa, query_paths, _ = get_xsim_multi(ref_filepath, query_filepath, features=features, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, metric=metric, num_paths=num_paths, enhance=True)
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
            "num_paths": num_paths
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
