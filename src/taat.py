import os
from pathlib import Path
import pprint
import tempfile
import numpy as np
import librosa
import soundfile as sf
from data_loader import *
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
        self.matrices = None
        self.paths = {}

    def plot(self):
        """
        Creates and displays a plot of the query result scores.
        """
        result = self.result
        scores = {}
        ref_files = []
        for (_, val) in result.items():
            score = val["score"]
            ref_file = val["collection_file"]
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
            ref_filepath = os.path.join(source_dir, match["collection_file"])
            matches_dir = make_dir_for_file_path(outdir, ref_filepath)
            ref_segs = match["collection_segments"]
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
                if (no_identity_match and filename != os.path.basename(query_filepath)) or \
                        (not no_identity_match):
                    ref_filepath = os.path.join(dirpath, filename)
                    ref_xsim, ref_rqa, ref_paths = run_backend(ref_filepath, ref_filepath,
                                                        backend=backend, features=features,
                                                        sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                        k=k, metric=metric, n_paths=n_paths,
                                                        enhance=True, zero_mean=zero_mean, n_filters=n_filters)
                    query_xsim, query_rqa, query_paths = run_backend(query_filepath, ref_filepath,
                                                            backend=backend, features=features,
                                                            sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                            k=k, metric=metric, n_paths=n_paths,
                                                            enhance=True, zero_mean=zero_mean, n_filters=n_filters)
                    paths, _ = get_time_formatted_paths(query_paths, n_fft=n_fft, hop_length=hop_length)
                    for (i, (ref_start, ref_stop, query_start, query_stop)) in enumerate(paths):
                        match = {
                            "score": get_path_score(ref_rqa, query_rqa, ref_paths[i], query_paths[i]),
                            "queryStart": query_start,
                            "queryStop": query_stop,
                            "collectionStart": ref_start,
                            "collectionStop": ref_stop,
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

def parse_query_output(query_filepath, query_output):
    result = {}
    keys = list(query_output.keys())
    for (i, v) in enumerate(list(query_output.values())):
        k = keys[i]
        score = float(np.mean([match["score"] for match in v]))
        #scores = [match["score"] for match in v]
        query_segs = [[match["queryStart"]*1000, match["queryStop"]*1000] for match in v]
        ref_segs = [[match["collectionStart"]*1000, match["collectionStop"]*1000] for match in v]
        result[f"results_{i}"] = {
            "score": score, #scores[0],
            "query_file": os.path.basename(query_filepath),
            "query_segments": query_segs,
            "collection_file": os.path.basename(k),
            "collection_segments": ref_segs
        }
    return result

def get_query_result(source_dir, query_filepath, sr=16000, chunk_length=30, overlap=0.5, features=["melspectrogram"],
                     n_fft=2048, hop_length=1024, k=5, metric="cosine", n_paths=5, enhance=True, zero_mean=False,
                     n_filters=5, no_identity_match=True):
    def check_identity_match(filename):
        return (no_identity_match and filename != os.path.basename(query_filepath) or (not no_identity_match))
    def print_fn(filename):
        return f"Computing cross-similarity for {os.path.basename(query_filepath)} against {os.path.basename(filename)}."
    matches = {}
    for (ref_filepath, ref_idx, ref_chunk) in walk(dir=source_dir, only_load_if=check_identity_match, print_fn=print_fn,
                                                   chunk_length=chunk_length, overlap=overlap, show_progress_bar=False):
        (ref_xsim, ref_rqa, ref_paths, _) = get_xsim_multi2(ref_chunk, ref_chunk,
                                                            features=features, sr=sr,
                                                            fft_size=n_fft, hop_length=hop_length,
                                                            k=k, metric=metric, n_paths=n_paths,
                                                            enhance=enhance, zero_mean=zero_mean, n_filters=n_filters)
        for (query_idx, query_chunk) in stream(query_filepath, chunk_length=chunk_length, overlap=overlap):
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
                    "collectionStart": ref_start,
                    "collectionStop": ref_stop,
                }
                if f"{ref_filepath} chunk_{ref_idx}" not in matches:
                    matches[f"{ref_filepath} chunk_{ref_idx}"] = [match]
                else:
                    matches[f"{ref_filepath} chunk_{ref_idx}"].append(match)
    return matches

def query2(source_dir, query_filepath, sr=16000, chunk_length=30, overlap=0.5, features=["melspectrogram"],
           n_fft=2048, hop_length=1024, k=3, metric="cosine", n_paths=5, pitch_shift=0,
           prune=False, score_threshold=0.25, path_margin=2, no_identity_match=True):
    query_filepath_original = query_filepath
    with tempfile.TemporaryDirectory() as tmpdir:
        query_filepath = write_pitch_shifted_file(input_filepath=query_filepath,
                                                  output_dir=tmpdir,
                                                  pitch_shift=pitch_shift)
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
        qr = QueryResult(query_filepath=query_filepath_original,
                         result={},
                         info=info)
        n_query_chunks = get_n_chunks(query_filepath, chunk_length, overlap)
        if (n_query_chunks==1):
            warning_msg = f"Skipping analysis of query file '{query_filepath_original}' with chunk_length={chunk_length} "\
                          f"and pitch_shift={pitch_shift}, as there are too few chunks to process "\
                           "(must be more than a single chunk)."
            print(warning_msg)
            return qr
        matches = get_query_result(source_dir=source_dir, query_filepath=query_filepath, sr=sr, chunk_length=chunk_length, overlap=overlap,
                                   features=features, n_fft=n_fft, hop_length=hop_length, k=k, metric=metric, n_paths=n_paths,
                                   enhance=True, zero_mean=prune, n_filters=5, no_identity_match=no_identity_match)
        parsed_result = parse_query_output2(query_filepath, matches, n_paths)
        mm = get_score_matrices(source_dir=source_dir,
                                query_filepath=query_filepath,
                                data=parsed_result.values(),
                                chunk_length=chunk_length,
                                overlap=overlap,
                                threshold=score_threshold,
                                no_identity_match=no_identity_match)
        qr._result = parsed_result
        qr.matrices = mm
        for i, (filepath, m) in enumerate(qr.matrices.items()):
            rqa, path = librosa.sequence.rqa(m)
            f = filter_result_for_path(filepath, qr._result.values(), path)
            merged = get_merged_path(f, chunk_length, overlap, path_margin)
            qr.result[f"results_{i}"] = merged
        return qr

def parse_seg_vals(filtered, key1, key2, s=1000, places=3):
    base_str = f"%.{places}f"
    return [[float(base_str % (d[key1]*s)), float(base_str % (d[key2]*s))] \
            for d in filtered]

def parse_query_output2(query_filepath, query_output, n_paths):
    temp = []
    query_filename = os.path.basename(query_filepath)
    for i, (key, entry) in enumerate(query_output.items()):
        for j in range(0, int(len(entry)/n_paths)):
            val = f"{query_filename} chunk_{j}"
            filtered = list(filter(lambda x: "query_file" in x and x["query_file"]==val, entry))
            score = float(np.mean([d["score"] for d in filtered]))
            #query_segs = [[d["queryStart"]*1000, d["queryStop"]*1000] for d in filtered]
            #ref_segs = [[d["collectionStart"]*1000, d["collectionStop"]*1000] for d in filtered]
            query_segs = parse_seg_vals(filtered, "queryStart", "queryStop")
            ref_segs = parse_seg_vals(filtered, "collectionStart", "collectionStop")
            temp.append({
                "score": score,
                "query_file": val,
                "query_segments": query_segs,
                "collection_file": key.replace("\\", "/") if os.name=="nt" else key,
                "collection_segments": ref_segs
            })
    results = {}
    for k, entry in enumerate(temp):
        results[f"results_{k}"] = entry
    return results

def get_score_matrix(data, dim, threshold=0.0):
    matrix = np.zeros(dim)
    for d in data:
        score = d["score"]
        if score > threshold:
            query_idx = int(d["query_file"].split(" chunk_")[-1])
            ref_idx = int(d["collection_file"].split(" chunk_")[-1])
            matrix[query_idx, ref_idx] = d["score"]
    return matrix

def get_score_matrices(source_dir, query_filepath, data, chunk_length,
                       overlap, threshold=0.0, no_identity_match=True):
    result = {}
    query_filename = os.path.basename(query_filepath)
    overlap_secs = int(chunk_length * overlap)
    d1 = math.ceil((librosa.get_duration(path=query_filepath)-overlap_secs) / overlap_secs)
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                if (no_identity_match and filename != query_filename) or (not no_identity_match):
                    ref_filepath = os.path.join(dirpath, filename)
                    d2 = math.ceil((librosa.get_duration(path=ref_filepath)-overlap_secs) / overlap_secs)
                    if d2 > 2:
                        f = list(filter(lambda x: os.path.basename(x["collection_file"].split(" chunk_")[0])==filename, data))
                        m = get_score_matrix(f, [d1, d2], threshold)
                        result[f"{source_dir}/{filename}"] = m
                    else:
                        warning_msg = f"Warning: skipping score matrix entry for collection file '{ref_filepath}', "\
                                       "as it contains too few columns (must be > 2)."
                        print(warning_msg)
    return result

def filter_result_for_path(filepath, data, path):
    result = []
    for [x, y] in path:
        f = filter(lambda d: int(d["query_file"].split(" chunk_")[-1])==x and \
                d["collection_file"].split(" chunk_")[0]==filepath and \
                int(d["collection_file"].split(" chunk_")[-1])==y, data)
        result = result + list(f)
    return result

def get_merged_path(path, chunk_length, overlap, margin):
    query_chunk_idxs = [int(d["query_file"].split(" chunk_")[-1]) for d in path]
    ref_chunk_idxs = [int(d["collection_file"].split(" chunk_")[-1]) for d in path]
    query_offsets = [float(idx * (chunk_length * overlap)) for idx in query_chunk_idxs]
    ref_offsets = [float(idx * (chunk_length * overlap)) for idx in ref_chunk_idxs]
    query_segs = np.array([d["query_segments"] for d in path])
    query_segs = [seg + query_offsets[i]*1000 for (i, seg) in enumerate(query_segs)]
    ref_segs = np.array([d["collection_segments"] for d in path])
    ref_segs = [seg + ref_offsets[i]*1000 for (i, seg) in enumerate(ref_segs)]
    ranges = zip_ranges(query_segs, ref_segs)
    merged = merge_ranges(ranges, margin)
    return {
        "query_file": path[0]["query_file"].split(" chunk_")[0],
        "collection_file": os.path.basename(path[0]["collection_file"]).split(" chunk_")[0],
        "query_segments": [[q1, q2] for [q1, q2, _, _] in merged],
        "collection_segments": [[r1, r2] for [_, _, r1, r2] in merged]
    }

def zip_ranges(qq, rr):
    result = []
    for (i, qs) in enumerate(qq):
        z = zip(qs, rr[i])
        zz = [[float(q1), float(q2), float(r1), float(r2)] \
                 for ((q1, q2),(r1, r2)) in z]
        result = result + zz
    return result

def onpick(event, matrix):
    x = int(np.round(event.xdata))
    y = int(np.round(event.ydata))
    ax = event.inaxes
    subfig = ax.get_figure()
    suptitle = subfig.get_suptitle()
    print(suptitle)
    suptitle = suptitle.split("Collection file: ")[-1].split("'")[1]
    title = ax.get_title()
    print(f"Matrix:, {title}")
    score = matrix[suptitle][x, y]
    print(f"Score for matrix coords {[x, y]}: {score}")

def plot_matrices(filepath, mm):
    if mm:
        keys = list(mm.keys())
        fig, axs = plt.subplots(nrows=len(mm), ncols=1, constrained_layout=True)
        if len(mm) == 1:
            axs = np.array([axs])
        plt.suptitle(f"Alignment scores for query file '{filepath}'")
        for ax in axs:
            ax.remove()
        gridspec = axs[0].get_subplotspec().get_gridspec()
        subfigs = [fig.add_subfigure(spec) for spec in gridspec]
        current_title = ""
        for row, subfig in enumerate(subfigs):
            m = mm[keys[row]]
            rqa, path = librosa.sequence.rqa(m, gap_onset=5, gap_extend=10)
            subfig.suptitle(f"Collection file: '{keys[row]}'")
            current_title = keys[row]
            ax = subfig.subplots(nrows=1, ncols=2)
            librosa.display.specshow(m, x_axis="frames", y_axis="frames", ax=ax[0])
            ax[0].set(title="Cross-similarity matrix")
            librosa.display.specshow(rqa, x_axis="frames", y_axis=f"frames", ax=ax[1])
            ax[1].set(title="Alignment score matrix")
            ax[1].plot(path[:, 1], path[:, 0], color="c")
            fig.canvas.current_title = row #current_title
        fig.canvas.mpl_connect("button_press_event", lambda e: onpick(e, mm))
        plt.show()

