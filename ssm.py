import numpy as np
from scipy import signal
#from numba import jit
import librosa
from matplotlib import pyplot as plt

    
# From libfmp/libfmp/c3/c3s1_post_processing.py
def smooth_downsample_feature_sequence(X, Fs, filt_len=41, down_sampling=10, w_type='boxcar'):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by convolution with a filter kernel
    """
    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature

# From libfmp/libfmp/c3/c3s1_post_processing.py
#@jit(nopython=True)
def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence"""
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm

#@jit(nopython=True)
def compute_sm_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product"""
    S = np.dot(np.transpose(X), Y)
    return S

#@jit(nopython=True)
def filter_diag_mult_sm(S, L=1, tempo_rel_set=np.asarray([1]), direction=0):
    """Path smoothing of similarity matrix by filtering in forward or backward direction"""
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((N, M))

    for s in range(0, num):
        M_ceil = int(np.ceil(M / tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1, M_ceil+1), M_ceil), M)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:, index_resample]

        S_L = np.zeros((N, M_ceil))
        S_extend_L = np.zeros((N + L, M_ceil + L))

        # Forward direction
        if direction == 0:
            S_extend_L[0:N, 0:M_ceil] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[pos:(N + pos), pos:(M_ceil + pos)]

        # Backward direction
        if direction == 1:
            S_extend_L[L:(N+L), L:(M_ceil+L)] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[(L-pos):(N + L - pos), (L-pos):(M_ceil + L - pos)]

        S_L = S_L / L
        resample = np.multiply(np.divide(np.arange(1, M+1), M), M_ceil)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)

    return S_L_final

# From libfmp/libfmp/c4/c4s2_ssm.py
#@jit(nopython=True)
def shift_cyc_matrix(X, shift=0):
    """Cyclic shift of features matrix along first dimension"""
    # Note: X_cyc = np.roll(X, shift=shift, axis=0) does to work for jit
    K, N = X.shape
    shift = np.mod(shift, K)
    X_cyc = np.zeros((K, N))
    X_cyc[shift:K, :] = X[0:K-shift, :]
    X_cyc[0:shift, :] = X[K-shift:K, :]
    return X_cyc

# From libfmp/libfmp/c4/c4s2_ssm.py
def compute_sm_ti(X, Y, L=1, tempo_rel_set=np.asarray([1]), shift_set=np.asarray([0]), direction=2):
    """Compute enhanced similaity matrix by applying path smoothing and transpositions"""
    for shift in shift_set:
        Y_cyc = shift_cyc_matrix(Y, shift)
        S_cyc = compute_sm_dot(X, Y_cyc)

        if direction == 0:
            S_cyc = filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=0)
        if direction == 1:
            S_cyc = filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=1)
        if direction == 2:
            S_forward = filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=0)
            S_backward = filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=1)
            S_cyc = np.maximum(S_forward, S_backward)
        if shift == shift_set[0]:
            S_TI = S_cyc
            I_TI = np.ones((S_cyc.shape[0], S_cyc.shape[1])) * shift
        else:
            # jit does not like the following lines
            # I_greater = np.greater(S_cyc, S_TI)
            # I_greater = (S_cyc > S_TI)
            I_TI[S_cyc > S_TI] = shift
            S_TI = np.maximum(S_cyc, S_TI)

    return S_TI, I_TI

# From libfmp/libfmp/c4/c4s2_threshold.py
def threshold_matrix(S, thresh, strategy='absolute', scale=False, penalty=0.0, binarize=False):
    """Treshold matrix in a relative fashion"""
    if np.min(S) < 0:
        raise Exception('All entries of the input matrix must be nonnegative')

    S_thresh = np.copy(S)
    N, M = S.shape
    num_cells = N * M

    if strategy == 'absolute':
        thresh_abs = thresh
        S_thresh[S_thresh < thresh] = 0

    if strategy == 'relative':
        thresh_rel = thresh
        num_cells_below_thresh = int(np.round(S_thresh.size*(1-thresh_rel)))
        if num_cells_below_thresh < num_cells:
            values_sorted = np.sort(S_thresh.flatten('F'))
            thresh_abs = values_sorted[num_cells_below_thresh]
            S_thresh[S_thresh < thresh_abs] = 0
        else:
            S_thresh = np.zeros([N, M])

    if strategy == 'local':
        thresh_rel_row = thresh[0]
        thresh_rel_col = thresh[1]
        S_binary_row = np.zeros([N, M])
        num_cells_row_below_thresh = int(np.round(M * (1-thresh_rel_row)))
        for n in range(N):
            row = S[n, :]
            values_sorted = np.sort(row)
            if num_cells_row_below_thresh < M:
                thresh_abs = values_sorted[num_cells_row_below_thresh]
                S_binary_row[n, :] = (row >= thresh_abs)
        S_binary_col = np.zeros([N, M])
        num_cells_col_below_thresh = int(np.round(N * (1-thresh_rel_col)))
        for m in range(M):
            col = S[:, m]
            values_sorted = np.sort(col)
            if num_cells_col_below_thresh < N:
                thresh_abs = values_sorted[num_cells_col_below_thresh]
                S_binary_col[:, m] = (col >= thresh_abs)
        S_thresh = S * S_binary_row * S_binary_col

    if scale:
        cell_val_zero = np.where(S_thresh == 0)
        cell_val_pos = np.where(S_thresh > 0)
        if len(cell_val_pos[0]) == 0:
            min_value = 0
        else:
            min_value = np.min(S_thresh[cell_val_pos])
        max_value = np.max(S_thresh)
        # print('min_value = ', min_value, ', max_value = ', max_value)
        if max_value > min_value:
            S_thresh = np.divide((S_thresh - min_value), (max_value - min_value))
            if len(cell_val_zero[0]) > 0:
                S_thresh[cell_val_zero] = penalty
        else:
            print('Condition max_value > min_value is voliated: output zero matrix')

    if binarize:
        S_thresh[S_thresh > 0] = 1
        S_thresh[S_thresh < 0] = 0
    return S_thresh

# From libfmp/libfmp/c4/c4s2_ssm.py
def compute_sm_from_filename(fn_wav, L=21, H=5, L_smooth=16, tempo_rel_set=np.array([1]),
                             shift_set=np.array([0]), strategy='relative', scale=True, thresh=0.15,
                             penalty=0.0, binarize=False):
    """Compute an SSM"""
    # Waveform
    Fs = 22050
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_duration = x.shape[0] / Fs

    # Chroma Feature Sequence and SSM (10 Hz)
    C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=2205, n_fft=4410)
    Fs_C = Fs / 2205

    # Chroma Feature Sequence and SSM
    X, Fs_feature = smooth_downsample_feature_sequence(C, Fs_C, filt_len=L, down_sampling=H)
    X = normalize_feature_sequence(X, norm='2', threshold=0.001)

    # Compute SSM
    S, I = compute_sm_ti(X, X, L=L_smooth, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
    S_thresh = threshold_matrix(S, thresh=thresh, strategy=strategy,
                                          scale=scale, penalty=penalty, binarize=binarize)
    return x, x_duration, X, Fs_feature, S_thresh, I

    