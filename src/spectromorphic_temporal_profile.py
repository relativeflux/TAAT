import copy
import numpy as np
import librosa
import scipy.stats
from scipy.spatial.distance import cdist


def get_chunk_profile(chunk, sr=16000):
    """
    Extracts a 47-dimensional structural signature combining 
    CQT-timbre textures and directional gesture morphing.
    """
    # 1. Timbral Texture via Constant-Q Transform (superior for noise/concrete audio)
    # 72 bins over 6 octaves perfectly maps the 0-8000Hz range at 16kHz
    try:
        cqt = np.abs(librosa.cqt(chunk, sr=sr, hop_length=256, n_bins=72, bins_per_octave=12))
        log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        # Extract CQT-MFCCs (Drop index 0 to eliminate steady-state loudness/hiss dependencies)
        mfcc = librosa.feature.mfcc(S=log_cqt, n_mfcc=14)[1:, :]
    except Exception:
        # Fallback to STFT-MFCC if a tiny chunk fails CQT boundary checks
        stft = np.abs(librosa.stft(chunk, n_fft=512, hop_length=256))
        log_stft = librosa.amplitude_to_db(stft, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_stft, n_mfcc=14)[1:, :]
    # Track how the timbre is evolving (Delta features)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    # 2. Extract Morphological Envelope Properties
    rms = librosa.feature.rms(y=chunk, frame_length=512, hop_length=128)[0]
    if np.max(rms) > 0:
        rms_norm = rms / (np.max(rms) + 1e-8)
        # Temporal Skewness: Distinguishes sharp attack/decay from long crescendos
        skew = scipy.stats.skew(rms_norm)
        # Crest Factor: Distinguishes sudden transient cracks from dense drone textures
        crest = np.max(rms_norm) / (np.mean(rms_norm) + 1e-8)
        # Temporal Centroid: Finds the temporal center of gravity of the gesture
        times = np.arange(len(rms_norm))
        temp_centroid = np.sum(times * rms_norm) / (np.sum(rms_norm) + 1e-8)
    else:
        skew, crest, temp_centroid = 0.0, 0.0, 0.0
    # 3. Concatenate all features into a static invariant footprint
    profile = np.hstack([
        np.mean(mfcc, axis=1),         # Average spectral coloration
        np.std(mfcc, axis=1),          # Textural instability/grain density
        np.mean(delta_mfcc, axis=1),   # Direction of spectral morphing
        np.std(delta_mfcc, axis=1),    # Velocity of spectral morphing
        skew, crest, temp_centroid     # Physical gesture envelope markers
    ])
    return profile

def get_gesture_pool(y, chunk_length=2.0, hop_length=0.5, sr=16000):
    chunk_samples = int(chunk_length * sr)
    hop_samples = int(hop_length * sr)
    profiles = []
    for start in range(0, len(y) - chunk_samples, hop_samples):
        chunk = y[start : start + chunk_samples]
        # Energy gate: Verify the chunk contains signal, not just empty space
        if np.max(np.abs(chunk)) > 0.001:
            profiles.append(get_chunk_profile(chunk, sr))
    return profiles

'''
You need to standardise the features or the similarity matrix to ensure scipy.spatial.distance.cdist only provides non-negative results to Librosa.
'''
def get_similarity_matrix(arr1, arr2, metric="cosine"):
    dist_matrix = scipy.spatial.distance.cdist(arr1, arr2, metric=metric)
    similarity = 1.0 - dist_matrix # Translate to similarity where 1.0 is a pure match
    # Prevent negative values from leaking into RQA
    return np.clip(similarity, 0.0, 1.0)

'''
Alternatively, if you want to preserve the inverse relationships captured by negative MFCC values (treating them as the lowest possible similarity rather than throwing them away), scale the output range from [-1, 1] down to [0, 1].
'''
def get_similarity_matrix2(arr1, arr2, metric="cosine"):
    dist_matrix = scipy.spatial.distance.cdist(arr1, arr2, metric=metric)
    similarity = 1.0 - dist_matrix
    # Shifts range from [-1, 1] to [0, 1]
    return (similarity + 1.0) / 2.0

def get_similarity_matrix3(arr1, arr2, metric="cosine", threshold_percentile=90):
    # 1. Get raw distance and convert to similarity range [-1, 1]
    dist_matrix = scipy.spatial.distance.cdist(arr1, arr2, metric=metric)
    similarity = 1.0 - dist_matrix
    # 2. Rescale range safely to [0, 1] to prevent the underflow bug
    similarity = (similarity + 1.0) / 2.0
    # 3. CRITICAL: Zero-out weak similarities to force paths to break
    # Only keep similarities in the top X% (e.g., top 10% if percentile=90)
    thresh = np.percentile(similarity, threshold_percentile)
    # Subtract threshold and clip at 0.0
    # Values below thresh become exactly 0.0; values above remain positive
    sparse_similarity = np.maximum(0.0, similarity - thresh)
    return sparse_similarity

def get_similarity_matrix4(arr1, arr2, metric="cosine"):
    # 1. Compute distances via SciPy
    dist_matrix = scipy.spatial.distance.cdist(arr1, arr2, metric=metric)
    similarity = 1.0 - dist_matrix
    # 2. Hard-force negative thresholds to zero
    similarity[similarity <= 0.0] = 0.0
    # 3. CRITICAL: Force a clean C-contiguous layout in memory.
    # This restructures the underlying RAM bytes into a strict row-major format,
    # ensuring Librosa's compiled Numba pointers read the bounds flawlessly.
    return np.ascontiguousarray(similarity, dtype=np.float32)

def apply_bidirectional_consensus_filter(sim_matrix, k=0.3):
    # --- BIDIRECTIONAL CONSENSUS FILTERING ---
    # Find best matches mapping A -> B and B -> A
    best_B_for_A = np.argmax(sim_matrix, axis=1)
    max_sim_A = np.max(sim_matrix, axis=1)
    best_A_for_B = np.argmax(sim_matrix, axis=0)
    # A match is ONLY valid if chunk A and chunk B mutually select each other
    mutual_matches = []
    path = []
    for idx_A, idx_B in enumerate(best_B_for_A):
        if best_A_for_B[idx_B] == idx_A:
            mutual_matches.append(max_sim_A[idx_A])
            path.append([idx_A, idx_B])
    if len(mutual_matches) == 0:
        return 0.0
    # Isolate only the highest quality core mutual agreements
    mutual_matches = np.sort(mutual_matches)
    top_k = max(1, int(len(mutual_matches) * k)) # Evaluate the top k% of agreements
    score = np.mean(mutual_matches[-top_k:])
    return float(np.clip(score, 0.0, 1.0)), path

def prune_rqa_path(path):
    underflow_mask = 18446744073709551614
    if path.dtype == np.uint64 or np.any(path == underflow_mask):
        path = path.astype(np.int64)
        path[path == underflow_mask] = -2
    valid_rows = (path[:, 0] >= 0) & (path[:, 1] >= 0) & (path[:, 0] < underflow_mask)
    return path[valid_rows]

def get_stp_match(filepath1, filepath2, sr=16000, chunk_length=2.0, hop_length=0.5,
                  n_paths=5, enhance=False, n_filters=5, zero_mean=False):
    """
    Compares two audio clips using spectromorphic temporal profiling. Uses
    bidirectional consensus matching over sliding structural windows. Returns a
    highly filtered match score between 0.0 and 1.0.
    """
    y1, _ = librosa.load(filepath1, sr=sr, mono=True)
    y2, _ = librosa.load(filepath2, sr=sr, mono=True)
    # Process Clip A into overlapping gesture pools
    profiles1 = get_gesture_pool(y1, chunk_length, hop_length, sr)
    # Process Clip B into overlapping gesture pools
    profiles2 = get_gesture_pool(y2, chunk_length, hop_length, sr)
    # Fallback to absolute zero match if either track is structurally silent
    if len(profiles1) == 0 or len(profiles2) == 0:
        return 0.0
    arr1 = np.array(profiles1)
    arr2 = np.array(profiles2)
    # Joint Z-Score Normalisation to equalise standard variance anomalies
    combined = np.vstack([arr1, arr2])
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0) + 1e-8
    arr_a_norm = (arr1 - mean) / std
    arr_b_norm = (arr2 - mean) / std
    # Get Cosine Similarity Matrix 
    sim_matrix = get_similarity_matrix(arr_a_norm, arr_b_norm)
    if enhance:
        sim_matrix = librosa.segment.path_enhance(sim_matrix, 64, n_filters=n_filters, zero_mean=zero_mean)
    # Apply bidirectional consensus filtering
    score, _ = apply_bidirectional_consensus_filter(sim_matrix)
    rqa_orig = librosa.sequence.rqa(sim_matrix)
    path = prune_rqa_path(rqa_orig[1])
    sim_matrix_copy = copy.deepcopy(sim_matrix)
    paths = []
    paths.append(path)
    path_idx = 0
    while path_idx < n_paths-1:
        for (i, j) in paths[path_idx]:
            sim_matrix_copy[i, j] = 0.0
        rqa = librosa.sequence.rqa(sim_matrix_copy)
        path = prune_rqa_path(rqa[1])
        paths.append(path)
        path_idx += 1
    return score, sim_matrix, rqa_orig[0], paths

def get_stp_match_v2(y1, y2, sr=16000, chunk_length=2.0, hop_length=0.5,
                     n_paths=5, enhance=False, n_filters=5, zero_mean=False):
    """
    Compares two audio clips using spectromorphic temporal profiling. Uses
    bidirectional consensus matching over sliding structural windows. Returns a
    highly filtered match score between 0.0 and 1.0.
    """
    # Process Clip A into overlapping gesture pools
    profiles1 = get_gesture_pool(y1, chunk_length, hop_length, sr)
    # Process Clip B into overlapping gesture pools
    profiles2 = get_gesture_pool(y2, chunk_length, hop_length, sr)
    # Fallback to absolute zero match if either track is structurally silent
    if len(profiles1) == 0 or len(profiles2) == 0:
        return 0.0
    arr1 = np.array(profiles1)
    arr2 = np.array(profiles2)
    # Joint Z-Score Normalisation to equalise standard variance anomalies
    combined = np.vstack([arr1, arr2])
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0) + 1e-8
    arr_a_norm = (arr1 - mean) / std
    arr_b_norm = (arr2 - mean) / std
    # Get Cosine Similarity Matrix 
    sim_matrix = get_similarity_matrix(arr_a_norm, arr_b_norm)
    if enhance:
        sim_matrix = librosa.segment.path_enhance(sim_matrix, 64, n_filters=n_filters, zero_mean=zero_mean)
    # Apply bidirectional consensus filtering
    score, _ = apply_bidirectional_consensus_filter(sim_matrix)
    rqa_orig = librosa.sequence.rqa(sim_matrix)
    path = prune_rqa_path(rqa_orig[1])
    sim_matrix_copy = copy.deepcopy(sim_matrix)
    paths = []
    paths.append(path)
    path_idx = 0
    while path_idx < n_paths-1:
        for (i, j) in paths[path_idx]:
            sim_matrix_copy[i, j] = 0.0
        rqa = librosa.sequence.rqa(sim_matrix_copy)
        path = prune_rqa_path(rqa[1])
        paths.append(path)
        path_idx += 1
    return score, sim_matrix, rqa_orig[0], paths

def get_time_formatted_stp_paths(paths, sr=16000, hop_length=0.5):
    overlap = 1 / hop_length
    paths_ = []
    for path in paths:
        query_start, ref_start = path[0]
        query_stop, ref_stop = path[-1]
        ref_start = (ref_start * sr) / overlap
        ref_stop = (ref_stop * sr) / overlap
        query_start = (query_start * sr) / overlap
        query_stop = (query_stop * sr) / overlap
        paths_.append([ref_start, ref_stop, query_start, query_stop])
    durs = [(r-p, s-q) for [p, r, q, s] in paths_]
    return paths_, durs

def get_subpaths(path, margin=3):
    result = []
    for [x, y] in path:
        if not result:
            result.append([[x, y]])
        else:
            last_elt = result[-1][-1]
            #if ((x>last_elt[0]) and (x-margin<=last_elt[0])) and \
               #((y>last_elt[1]) and (y-margin<=last_elt[1])):
            #if (x-margin<=last_elt[0]) and (y-margin<=last_elt[1]):
            if (abs(x-last_elt[0])<=margin) and (abs(y-last_elt[1])<=margin):
                result[-1].append([x, y])
            else:
                result.append([[x, y]])
    return [np.array(elt) for elt in result]

'''
def test(filepath1, filepath2, sr=16000, features=["melspectrogram"],
         n_fft=2048, hop_length=1024, k=3, enhance=True):
    y1, _ = librosa.load(filepath1, sr=sr, mono=True)
    y2, _ = librosa.load(filepath2, sr=sr, mono=True)
    f1 = apply_features(y1, features=features, sr=sr, n_fft=n_fft, hop_length=hop_length)
    f2 = apply_features(y2, features=features, sr=sr, n_fft=n_fft, hop_length=hop_length)
    f1 = librosa.feature.stack_memory(f1, n_steps=10, delay=3)
    f2 = librosa.feature.stack_memory(f2, n_steps=10, delay=3)
    sim_matrix = librosa.segment.cross_similarity(f1, f2, k=k, metric="cosine", mode="affinity")
    if enhance:
        sim_matrix = librosa.segment.path_enhance(sim_matrix, 64, n_filters=5)
    return apply_bidirectional_consensus_filter(sim_matrix)

def test_stp(dir, query_filepath):
    query_filename = os.path.basename(query_filepath)
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith("wav"):
                ref_filepath = os.path.join(dirpath, filename)
                score, _, _, _ = (get_stp_match(query_filepath, ref_filepath))
                ref_filename = os.path.basename(ref_filepath)
                print(f"MTP match score for {ref_filename} against {query_filename} is: {score}.")

def test3(a):
    best_B_for_A = np.argmax(a, axis=1)
    max_sim_A = np.max(a, axis=1)
    best_A_for_B = np.argmax(a, axis=0)
    mutual_matches = []
    path = []
    for idx_A, idx_B in enumerate(best_B_for_A):
        if best_A_for_B[idx_B] == idx_A:
            mutual_matches.append(max_sim_A[idx_A])
            path.append([idx_A, idx_B])
    return mutual_matches, path
'''

