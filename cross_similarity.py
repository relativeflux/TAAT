import math
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.spatial.distance as distance
import sklearn


def butter_bandpass_filter(data, lowcut=180, highcut=3000, sr=16000, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.lfilter(b, a, data)

def stft(y, sr=22050, fft_size=2048, hop_length=1024):
    spect = librosa.stft(y, n_fft=fft_size, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(spect), ref=np.max)

def cqt(y, sr=22050, fft_size=2048, hop_length=1024):
    spect = librosa.cqt(y, sr=sr, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(spect), ref=np.max)

def melspectrogram(y, sr=22050, fft_size=2048, hop_length=1024):
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft_size, hop_length=hop_length)
    return librosa.amplitude_to_db(mel_spect, ref=np.max)

def mfcc(y, sr=22050, fft_size=2048, hop_length=1024):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

def spectral_centroid(y, sr=22050, fft_size=2048, hop_length=1024):
    return librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=fft_size, hop_length=hop_length)

def spectral_bandwidth(y, sr=22050, fft_size=2048, hop_length=1024):
    return librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=fft_size, hop_length=hop_length)

def spectral_contrast(y, sr=22050, fft_size=2048, hop_length=1024):
    return librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=fft_size, hop_length=hop_length)

def spectral_flatness(y, sr=22050, fft_size=2048, hop_length=1024):
    return librosa.feature.spectral_flatness(y=y, n_fft=fft_size, hop_length=hop_length)

args = locals()

def get_xsim(y_comp, y_ref, sr=16000, feature="melspectrogram", fft_size=2048, hop_length=1024, k=2, metric='euclidean', mode='affinity', gap_onset=np.inf, gap_extend=np.inf, knight_moves=False):
    ref = args[feature](y_ref, sr=sr, fft_size=fft_size, hop_length=hop_length)
    comp = args[feature](y_comp, sr=sr, fft_size=fft_size, hop_length=hop_length)
    x_ref = librosa.feature.stack_memory(ref, n_steps=10, delay=3)
    x_comp = librosa.feature.stack_memory(comp, n_steps=10, delay=3)
    xsim = librosa.segment.cross_similarity(x_comp, x_ref, k=k, metric=metric, mode=mode)
    rqa = librosa.sequence.rqa(xsim, gap_onset=gap_onset, gap_extend=gap_extend, knight_moves=knight_moves)
    return xsim, rqa

def get_xsim_start_times(rqa, y_ref, y_comp, sr):
    score, plot = rqa
    score1_len, score2_len = score.shape
    plot_start1, plot_start2 = plot[0]
    start1_pcnt = plot_start1 / score1_len
    start2_pcnt = plot_start2 / score2_len
    start1_sample = math.floor(len(y_ref) * start1_pcnt)
    start2_sample = math.floor(len(y_comp) * start2_pcnt)
    return (librosa.samples_to_time(start1_sample),
            librosa.samples_to_time(start2_sample))

def get_xsim_start_end_times(rqa, y_ref, y_comp, sr):
    score, plot = rqa
    score1_len, score2_len = score.shape
    # Compute start times...
    plot_start1, plot_start2 = plot[0]
    start1_pcnt = plot_start1 / score1_len
    start2_pcnt = plot_start2 / score2_len
    start1_sample = math.floor(len(y_ref) * start1_pcnt)
    start2_sample = math.floor(len(y_comp) * start2_pcnt)
    # Compute end times...
    plot_end1, plot_end2 = plot[-1]
    end1_pcnt = plot_end1 / score1_len
    end2_pcnt = plot_end2 / score2_len
    end1_sample = math.floor(len(y_ref) * end1_pcnt)
    end2_sample = math.floor(len(y_comp) * end2_pcnt)
    # Convert samples to times in seconds and return...
    return (np.round(librosa.samples_to_time(start1_sample), 2),
            np.round(librosa.samples_to_time(end1_sample), 2),
            np.round(librosa.samples_to_time(start2_sample), 2),
            np.round(librosa.samples_to_time(end2_sample), 2))

'''
def plot_xsim(xsim, rqa, hop_length):
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    imgsim = librosa.display.specshow(xsim, x_axis='s', y_axis='s', hop_length=hop_length, ax=ax[0])
    ax[0].set(title='Cross-similarity matrix')
    imgrqa = librosa.display.specshow(rqa, x_axis='s', y_axis='s', cmap='magma_r', hop_length=hop_length, ax=ax[1])
    ax[1].set(title='RQA')
    ax[1].label_outer()
    fig.colorbar(imgsim, ax=ax[0], orientation='horizontal', ticks=[0, 1])
    fig.colorbar(imgrqa, ax=ax[1], orientation='horizontal')
    plt.show()
'''

def plot_xsim(xsim, rqa, start1, end1, start2, end2):
    score, path = rqa
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    librosa.display.specshow(xsim, x_axis='frames', y_axis='frames', ax=ax[0])
    ax[0].set(title='Cross-similarity matrix')
    librosa.display.specshow(score, x_axis='frames', y_axis='frames', ax=ax[1])
    ax[1].set(title='Alignment score matrix')
    ax[1].plot(path[:, 1], path[:, 0], label='Optimal path', color='c')
    ax[1].legend()
    ax[1].label_outer()
    start_end_data = f'Start 1: {start1}, End 1: {end1} | Start 2: {start2}, End 2: {end2}'
    plt.figtext(0, 0, start_end_data, va='top')
    plt.show(block=False)

'''
import librosa
from cross_similarity import get_xsim_multi, plot_xsim_multi, write_path_files
filepath1 = '../Dropbox/Miscellaneous/TAAT/Data/sg-audio-datasets/02 c-in your image.wav'
filepath2 = '../Dropbox/Miscellaneous/TAAT/Data/sg-audio-datasets/02 i-no fish.wav'

xsim, rqa, paths, info = get_xsim_multi(filepath1, filepath2, feature='melspectrogram', fft_size=8192, hop_length=8192, k=20, metric='cosine', gap_onset=5, gap_extend=10, knight_moves=True, num_paths=10)

file_path3 = '../Dropbox/Miscellaneous/TAAT/Data/Test Cases/Test 4/data/003 Chord composition V (op.8).wav'
file_path4 = '../Dropbox/Miscellaneous/TAAT/Data/Test Cases/Test 4/input/023 Daguerreo types, Op. 32B.wav'
'''

###################################################################

import copy
from matplotlib import collections
import soundfile as sf
import os
import json
import stumpy

def get_xsim_multi(y_comp_path, y_ref_path, sr=16000, feature="melspectrogram", fft_size=2048, hop_length=2048, k=2, metric="cosine", mode="affinity", gap_onset=np.inf, gap_extend=np.inf, knight_moves=False, num_paths=5, lowcut=180, highcut=3000, norm=False, enhance=False):
    y_ref, _ = librosa.load(y_ref_path, sr=sr, mono=True)
    y_comp, _ = librosa.load(y_comp_path, sr=sr, mono=True)
    y_ref = butter_bandpass_filter(y_ref, lowcut=lowcut, highcut=highcut, sr=sr)
    y_comp = butter_bandpass_filter(y_comp, lowcut=lowcut, highcut=highcut, sr=sr)
    ref = args[feature](y_ref, sr=sr, fft_size=fft_size, hop_length=hop_length)
    comp = args[feature](y_comp, sr=sr, fft_size=fft_size, hop_length=hop_length)
    if norm:
        ref = stumpy.core.z_norm(ref)
        comp = stumpy.core.z_norm(comp)
    x_ref = librosa.feature.stack_memory(ref, n_steps=10, delay=3)
    x_comp = librosa.feature.stack_memory(comp, n_steps=10, delay=3)
    xsim_orig = librosa.segment.cross_similarity(x_comp, x_ref, k=k, metric=metric, mode=mode)
    if enhance:
        xsim_orig = librosa.segment.path_enhance(xsim_orig, 64, n_filters=10)
    rqa_orig = librosa.sequence.rqa(xsim_orig, gap_onset=gap_onset, gap_extend=gap_extend, knight_moves=knight_moves)
    xsim_copy = copy.deepcopy(xsim_orig)
    paths = []
    paths.append(rqa_orig[1])
    path_idx = 0
    while path_idx < num_paths-1:
        for (i, j) in paths[path_idx]:
            xsim_copy[i, j] = 0.0
        rqa = librosa.sequence.rqa(xsim_copy, gap_onset=gap_onset, gap_extend=gap_extend, knight_moves=knight_moves)
        paths.append(rqa[1])
        path_idx += 1
    info = {
        "feature": feature,
        "sample_rate": sr,
        "fft_size": fft_size,
        "hop_length": hop_length,
        "k": k,
        "metric": metric,
        "num_paths": num_paths,
        #"gap_onset": gap_onset,
        #"gap_extend": gap_extend,
        "knight_moves": knight_moves,
        #"z_normed": norm,
        #"y_ref_path": y_ref_path,
        #"y_comp_path": y_comp_path,
        "path_length": len(paths[0])
    }
    return xsim_orig, rqa_orig[0], paths, info

## Helper function for quickly getting properly time-formatted paths:
def get_time_formatted_paths(paths, n_fft=2048, hop_length=1024):
    paths_ = []
    for path in paths:
        ref_start, query_start = path[0]
        ref_stop, query_stop = path[-1]
        ref_start = float(librosa.frames_to_time(ref_start, n_fft=n_fft, hop_length=hop_length))
        ref_stop = float(librosa.frames_to_time(ref_stop, n_fft=n_fft, hop_length=hop_length))
        query_start = float(librosa.frames_to_time(query_start, n_fft=n_fft, hop_length=hop_length))
        query_stop = float(librosa.frames_to_time(query_stop, n_fft=n_fft, hop_length=hop_length))
        paths_.append([ref_start, ref_stop, query_start, query_stop])
    durs = [(r-p, s-q) for [p, r, q, s] in paths_]
    return paths_, durs

import skimage
from hog_descriptor import chi2_distance

def get_hog_descriptor(img_data, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), binarize=True):
    fd, hog = skimage.feature.hog(img_data,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        channel_axis=-1,
    )
    if binarize:
        thresh = skimage.filters.threshold_otsu(hog)
        hog = hog > thresh
    return fd, hog

def frobenius_inner_product(a, b):
    return np.trace(np.matmul(a.T, b))

def frobenius_distance(a, b, norm=True):
    dist = np.linalg.norm(a - b, "fro")
    if norm:
        return dist / np.linalg.norm(a, "fro")
    else:
        return dist

def get_path_data(rqa, path):
    return [float(rqa[i,j]) for (i,j) in path]

def get_path_distance(rqa1, rqa2, path1, path2):
    a = get_path_data(rqa1, path1)
    b = get_path_data(rqa2, path2)
    a, b = sorted([a, b], key=lambda arr: len(arr))
    pad_len = abs(len(b) - len(a))
    a = np.pad(a, (0, pad_len), "constant")
    return distance.cosine(a, b)

def get_path_score(rqa1, rqa2, path1, path2):
    a = get_path_data(rqa1, path1)
    b = get_path_data(rqa2, path2)
    a, b = sorted([a, b], key=lambda arr: len(arr))
    pad_len = abs(len(b) - len(a))
    a = np.pad(a, (0, pad_len), "constant")
    dist = sklearn.metrics.pairwise.cosine_similarity([a], [b])
    return float(dist[0][0])

def get_rqa_score(ref_rqa, query_rqa):
    max_ref_rqa = np.max(ref_rqa)
    max_query_rqa = np.max(query_rqa)
    return max_query_rqa / max_ref_rqa

def get_rqa_score2(ref_rqa, query_rqa, query_paths, threshold=0.25):
    max_ref_rqa = np.max(ref_rqa)
    res = []
    for path in query_paths:
        m, n = path[-1]
        curr_score = float(query_rqa[m,n] / max_ref_rqa)
        if (curr_score>threshold):
            res.append(curr_score)
    return len(res) / len(query_paths)

def query(query_filepath, source_dir, sr=16000, n_fft=2048, hop_length=1024, verbose=True, no_identity_match=True, k=5, num_paths=5, enhance=True):
    matches = {}
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                ref_filepath = os.path.join(dirpath, filename)
                if no_identity_match==True and os.path.basename(ref_filepath) != os.path.basename(query_filepath):
                    ref_xsim, ref_rqa, ref_paths, _ = get_xsim_multi(ref_filepath, ref_filepath, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, num_paths=num_paths, enhance=enhance)
                    print(f"Computing cross-similarity for {os.path.basename(query_filepath)} against {os.path.basename(ref_filepath)}.")
                    query_xsim, query_rqa, query_paths, _ = get_xsim_multi(ref_filepath, query_filepath, sr=sr, fft_size=n_fft, hop_length=hop_length, k=k, num_paths=num_paths, enhance=enhance)
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
    if verbose:
        return matches
    else:
        return parse_query_output(query_filepath, matches)

def parse_query_output(query_filepath, query_output):
    result = {}
    keys = list(query_output.keys())
    for (i, v) in enumerate(list(query_output.values())):
        k = keys[i]
        score = float(np.mean([match["score"] for match in v]))
        query_segs = [[match["queryStart"]*1000, match["queryStop"]*1000] for match in v]
        ref_segs = [[match["referenceStart"]*1000, match["referenceStop"]*1000] for match in v]
        result[f"results_{i}"] = {
            "score": score,
            "query_file": os.path.basename(query_filepath),
            "query_segments": query_segs,
            "reference_file": os.path.basename(k),
            "reference_segments": ref_segs
        }
    return result

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)

def plot_xsim_multi(xsim, rqa, paths):
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    librosa.display.specshow(xsim, x_axis='frames', y_axis='frames', ax=ax[0])
    ax[0].set(title='Cross-similarity matrix')
    librosa.display.specshow(rqa, x_axis='frames', y_axis='frames', ax=ax[1])
    ax[1].set(title='Alignment score matrix')
    for path in paths:
        ax[1].plot(path[:, 1], path[:, 0], color='c', picker=True)
    #fig.canvas.mpl_connect('pick_event', onpick)
    plt.show(block=False)

import pandas as pd
import io
import csv
import random
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

'''
def plot_xsim_parallel_coordinates(csv):
    csv = csv if (csv.split(".")[-1] == "csv") else io.StringIO(csv)
    df = pd.read_csv(csv)
    plt.figure()
    pd.plotting.parallel_coordinates(
        df[["feature", "sample_rate", "fft_size", "hop_length", "k", "metric", "num_paths", "gap_onset", "gap_extend", "knight_moves", "path_length"]], "feature")
    plt.show()
'''

def plot_xsim_parallel_coordinates(csv):
    csv = csv if (csv.split(".")[-1] == "csv") else io.StringIO(csv)
    df = pd.read_csv(csv)
    fig = px.parallel_coordinates(df,
        color="feature",
        dimensions=["feature", "fft_size", "hop_length", "k", "metric",  "knight_moves", "path_length"],
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2)
    #fig.update_layout(yaxis_type="category")
    fig.show()

def infos_to_csv(infos):
    output = io.StringIO()
    fieldnames = list(infos[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for info in infos:
        writer.writerow(info)
    return output.getvalue()

def test_xsim_multi(y_comp_path, y_ref_path, n=10, sr=16000, k_range=[2,20]):
    infos = []
    fft_sizes = [1024, 2048, 4096, 8192, 16384]
    features = ["stft", "melspectrogram", "cqt", "mfcc"]
    metrics = ["cosine", "euclidean", "manhattan"]
    for _ in range(0, n):
        fft_size = random.choice(fft_sizes)
        feature = random.choice(features)
        metric = random.choice(metrics)
        k = random.randint(k_range[0], k_range[1])
        knight_moves = random.choice([True, False])
        _, _, _, info = get_xsim_multi(y_comp_path, y_ref_path, sr=sr, feature=feature, fft_size=fft_size, hop_length=fft_size, metric=metric, k=k, knight_moves=knight_moves, num_paths=1)
        info["feature"] = features.index(feature)
        info["metric"] = metrics.index(metric)
        infos.append(info)
    csv_string = infos_to_csv(infos)
    plot_xsim_parallel_coordinates(csv_string)

def get_xsim_start_end_times2(score, plot, y_ref, y_comp, sr):
    score1_len, score2_len = score.shape
    # Compute start times...
    plot_start1, plot_start2 = plot[0]
    start1_pcnt = plot_start1 / score1_len
    start2_pcnt = plot_start2 / score2_len
    start1_sample = math.floor(len(y_ref) * start1_pcnt)
    start2_sample = math.floor(len(y_comp) * start2_pcnt)
    # Compute end times...
    plot_end1, plot_end2 = plot[-1]
    end1_pcnt = plot_end1 / score1_len
    end2_pcnt = plot_end2 / score2_len
    end1_sample = math.floor(len(y_ref) * end1_pcnt)
    end2_sample = math.floor(len(y_comp) * end2_pcnt)
    # Convert samples to times in seconds and return...
    return [np.round(librosa.samples_to_time(start1_sample), 2),
            np.round(librosa.samples_to_time(end1_sample), 2),
            np.round(librosa.samples_to_time(start2_sample), 2),
            np.round(librosa.samples_to_time(end2_sample), 2)]

def write_path_file(outdir, file_path, filename_prefix, start, end, sr):
    seg, _ = librosa.load(file_path, sr=sr, mono=True, offset=start, duration=end-start)
    filename = f"{filename_prefix}.start={start}_end={end}.wav"
    sf.write(os.path.join(outdir, filename), seg, sr)

def write_path_files(outdir, score, paths, y_ref_path, y_comp_path, sr, info):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if info:
        with open(os.path.join(outdir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)
    y_ref, _ = librosa.load(y_ref_path, sr=sr, mono=True)
    y_comp, _ = librosa.load(y_comp_path, sr=sr, mono=True)
    p = [get_xsim_start_end_times2(score, path, y_ref, y_comp, sr) for path in paths]
    for (i, start_end_pair) in enumerate(p):
        start1, end1, start2, end2 = start_end_pair
        write_path_file(outdir, y_ref_path, f"y_ref_{i}", start1, end1, sr)
        write_path_file(outdir, y_comp_path, f"y_comp_{i}", start2, end2, sr)

def match_xsim_multi(paths, timings_file, diff=5):
    paths = sorted(paths)
    with open(timings_file) as f:
        content = json.load(f)
    timings = content['data']
    result = {}
    for i in range(0, len(timings)):
        result[i] = None
    for (i, [[a,b],[c,d]]) in enumerate(timings):
        for [aa,bb,cc,dd] in paths:
            diff_a = abs(a-aa)
            diff_b = abs(b-bb)
            diff_c = abs(c-cc)
            diff_d = abs(d-dd)
            if (diff_a<=diff and diff_b<=diff
                and diff_c<=diff and diff_d<=diff):
                result[i] = [aa,bb,cc,dd]
    return result

def match_xsim_multi2(paths, timings_file, diff=5):
    with open(timings_file) as f:
        content = json.load(f)
    timings = content['data']
    result = {}
    for i in range(0, len(timings)):
        result[i] = None
    for (i, [[a,b],[c,d]]) in enumerate(timings):
        for [aa,bb,cc,dd] in paths:
            if result[i]:
                [aaa,bbb,ccc,ddd] = result[i]
                diff_a = aaa-aa
                diff_b = bbb-bb
                diff_c = ccc-cc
                diff_d = ddd-dd
                if (diff_a<diff and diff_b<diff
                    and diff_c<diff and diff_d<diff):
                    result[i] = [aa,bb,cc,dd]
            else:
                diff_a = abs(a-aa)
                diff_b = abs(b-bb)
                diff_c = abs(c-cc)
                diff_d = abs(d-dd)
                if (diff_a<=diff and diff_b<=diff
                    and diff_c<=diff and diff_d<=diff):
                    result[i] = [aa,bb,cc,dd]
    return result


def match_xsim_multi3(paths, timings_file):
    with open(timings_file) as f:
        content = json.load(f)
    timings = content['data']
    result = {}
    for i in range(0, len(timings)):
        result[i] = None
    for (i, [[a,b],[c,d]]) in enumerate(timings):
        candidate = None
        for [aa,bb,cc,dd] in paths:
            if not candidate:
                candidate = [aa, bb, cc, dd]
            else:
                if abs(a-aa)<abs(a-candidate[0]) and abs(b-bb)<abs(b-candidate[1]) \
                   and abs(c-cc)<abs(c-candidate[2]) and abs(d-dd)<abs(d-candidate[3]):
                   candidate = [aa,bb,cc,dd]
        result[i] = candidate
    return result

def match_xsim_multi4(paths, timings_file, diff=5):
    paths = sorted(paths)
    with open(timings_file) as f:
        content = json.load(f)
    timings = content['data']
    result = []
    for [[a,b],[c,d]] in timings:
        candidate = paths[0]
        for [aa,bb,cc,dd] in paths[1:]:
            if (abs(a-candidate[0]) <= diff) and (abs(b-candidate[1]) <= diff) and (abs(c-candidate[2]) <= diff) and (abs(d-candidate[3]) <= diff):
                candidate = [aa,bb,cc,dd]
            #else:
                #candidate = None
        result.append(candidate)
    return result

def match_xsim_multi5(paths, timings_file, diff=5):
    with open(timings_file) as f:
        content = json.load(f)
    timings = content['data']
    result = {}
    for i in range(0, len(timings)):
        result[i] = None
    for (i, [[a,b],[c,d]]) in enumerate(timings):
        for [aa,bb,cc,dd] in paths:
            diff_a = abs(a-aa)
            diff_b = abs(b-bb)
            if (diff_a<=diff and diff_b<=diff):
                result[i] = [aa,bb,cc,dd]
    return result


