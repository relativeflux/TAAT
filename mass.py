import os
import numpy as np
import librosa
import stumpy
import matplotlib.pyplot as plt
import soundfile as sf
from peak_extraction import find_peaks


def get_matches(filepath1, filepath2, max_distance="auto"):
    _, ts = find_peaks(filepath1)
    _, query = find_peaks(filepath2)
    ts = np.array(sorted(ts)).astype(np.float64)
    query = np.array(sorted(query)).astype(np.float64)
    frames = ts[:,0]
    ts = ts[:,2]
    last_query_frame = query[-1][0]
    query = query[:,2]
    max_dist = lambda D: max(np.mean(D) - 4 * np.std(D), np.min(D))
    matches = stumpy.match(query, ts, max_distance=max_dist if max_distance=="auto" else None)
    return frames, query, matches, last_query_frame

def write_match_file(outdir, ts_file_path, filename_prefix, start, end, sr):
    seg, _ = librosa.load(ts_file_path, sr=sr, mono=True, offset=start, duration=end-start)
    filename = f"{filename_prefix}.start={start}_end={end}.wav"
    sf.write(os.path.join(outdir, filename), seg, sr)

def write_match_files(outdir, ts_file_path, matches, frames, last_query_frame, sr=16000, n_fft=2048, hop_length=1024):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for (i, (_, match_idx)) in enumerate(matches):
        start = librosa.frames_to_time(frames[match_idx], n_fft=n_fft, hop_length=hop_length)
        end = librosa.frames_to_time(frames[match_idx]+last_query_frame, n_fft=n_fft, hop_length=hop_length)
        write_match_file(outdir, ts_file_path, f"match_{i}", start, end, sr)

def plot_matches(ts, query, matches):
    Q_z_norm = stumpy.core.z_norm(query)
    plt.suptitle('Comparing The Query To All Matches (Default max_distance)', fontsize='30')
    plt.xlabel('Time', fontsize ='20')
    plt.ylabel('Acceleration', fontsize='20')
    for match_distance, match_idx in matches:
        match_z_norm = stumpy.core.z_norm(ts[match_idx:match_idx+len(query)])
        plt.plot(match_z_norm, lw=2)
    plt.plot(Q_z_norm, lw=4, color="black", label="Query Subsequence, Q_df")
    plt.legend()
    plt.show()

def get_motifs(ts_filepath, query_filepath, sr=16000, n_fft=1024, hop_length=1024, m=500, channel=16):
    ts, _ = librosa.load(ts_filepath, sr=sr, mono=True)
    query, _ = librosa.load(query_filepath, sr=sr, mono=True)
    ts = ts.astype(np.float64)
    query = query.astype(np.float64)
    ts_mfcc = librosa.feature.mfcc(y=ts, sr=sr, n_fft=n_fft, hop_length=hop_length)[channel]
    query_mfcc = librosa.feature.mfcc(y=query, sr=sr, n_fft=n_fft, hop_length=hop_length)[channel]
    return stumpy.stump(T_A = ts_mfcc, m = m, T_B = query_mfcc, ignore_trivial = False)

def plot_best_motif_match(ts_mfcc, query_mfcc, motifs, m):
    best_motif_index = motifs[:, 0].argmin()
    ts_z_norm_motif = stumpy.core.z_norm(ts_mfcc[best_motif_index : best_motif_index + m])
    query_motif_index = motifs[best_motif_index, 1]
    query_z_norm_motif = stumpy.core.z_norm(query_mfcc[query_motif_index:query_motif_index+m])
    plt.plot(ts_z_norm_motif, label='ts')
    plt.plot(query_z_norm_motif, label='query')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def write_best_motif_file(outdir, ts_filepath, query_filepath, motifs, m=500, sr=16000, n_fft=1024, hop_length=1024, channel=16):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ts_filename = os.path.basename(ts_filepath).split(".wav")[0]
    query_filename = os.path.basename(query_filepath).split(".wav")[0]
    subfolder_name = f"{ts_filename}_&_{query_filename}"
    subfolder_path = os.path.join(outdir, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    ts, _ = librosa.load(ts_filepath, sr=sr, mono=True)
    query, _ = librosa.load(query_filepath, sr=sr, mono=True)
    ts_mfcc = librosa.feature.mfcc(y=ts, sr=sr, n_fft=n_fft, hop_length=hop_length)[channel]
    query_mfcc = librosa.feature.mfcc(y=query, sr=sr, n_fft=n_fft, hop_length=hop_length)[channel]
    ts_motif_index = motifs[:, 0].argmin()
    query_motif_index = motifs[ts_motif_index, 1]
    ts_start = librosa.frames_to_time(ts_motif_index, n_fft=n_fft, hop_length=hop_length)
    ts_end = librosa.frames_to_time(ts_motif_index + m, n_fft=n_fft, hop_length=hop_length)
    write_match_file(subfolder_path, ts_filepath, f"ts_motif", ts_start, ts_end, sr)
    query_start = librosa.frames_to_time(query_motif_index, n_fft=n_fft, hop_length=hop_length)
    query_end = librosa.frames_to_time(query_motif_index + m, n_fft=n_fft, hop_length=hop_length)
    write_match_file(subfolder_path, query_filepath, f"query_motif", query_start, query_end, sr)
