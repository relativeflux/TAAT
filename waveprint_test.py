import math
from obspy import read
import numpy as np
from obspy.core import UTCDateTime
import time
import datetime
import librosa
import pywt
from PIL import Image
from sklearn.preprocessing import normalize
import json
from scipy.signal import spectrogram


class WaveprintTest():

    def __init__(self, sample_rate=100, window_length=6.0, window_lag=0.2, fingerprint_length=128, fingerprint_lag=10, min_freq=0, max_freq=None, nfreq=32, ntimes=64, partition_length=28800):
        self.nwindows = None
        self.d1 = None
        self.d2 = None
        self.new_d1 = int(nfreq)
        self.new_d2 = int(ntimes)
        self.fp_len = fingerprint_length
        self.fp_lag = fingerprint_lag
        self.spec_len= 6.0
        self.spec_lag = 0.2
        self.window_len = window_length
        self.window_lag = window_lag
        self.sample_rate = sample_rate
        self.min_freq = min_freq #2.0
        self.max_freq = max_freq or sample_rate/2.0
        self.frequencies = None
        self.times = None
        self.partition_len = partition_length

    def get_window_params(self, N, L, dL):
        idx0 = np.asarray(range(0, N+1, dL))
        idx2 = np.asarray(range(L,N+1,dL))
        nWindows = len(idx2)
        idx1 = idx0[0:nWindows]
        return nWindows, idx1, idx2

    def resize_spectral_images(self, spectral_images, new_d1, new_d2):
        new_spectral_images = np.zeros([self.nwindows,self.new_d1,self.new_d2])
        for i in range(self.nwindows):
            new_spectral_images[i,:,:] = np.array(Image.fromarray(spectral_images[i,:,:]).resize(size=(new_d2, new_d1), resample=Image.Resampling.BILINEAR))
        return new_spectral_images

    def unwrap_wavelet_coeffs(self, coeffs):
        L = len(coeffs)
        cA = coeffs[0]
        for i in range(1,L):
            (cH, cV, cD) = coeffs[i]
            cA = np.concatenate((np.concatenate((cA, cV),axis= 1),np.concatenate((cH, cD),axis = 1)),axis=0)
        return cA

    def spectrogram_to_spectral_images(self, Sxx):
        nFreq, nTimes = np.shape(Sxx)
        nWindows, idx1, idx2 = self.get_window_params(nTimes, self.fp_len, self.fp_lag)
        spectral_images = np.zeros([nWindows, nFreq, self.fp_len])
        for i in range(nWindows):
            spectral_images[i,:,:] = Sxx[:,idx1[i]:idx2[i]]
        self.nwindows = nWindows
        nWindows, self.d1, self.d2 = np.shape(spectral_images)
        return spectral_images, nWindows, idx1, idx2

    def spectral_images_to_wavelet(self, spectral_images, wavelet = pywt.Wavelet('db1')):
        if (int(self.new_d1)!=self.d1) or (int(self.new_d2)!=self.d2):
            spectral_images = self.resize_spectral_images(spectral_images, self.new_d1, self.new_d2)
        haar_images = np.zeros([self.nwindows,self.new_d1,self.new_d2])
        for i in range(self.nwindows):
            coeffs = pywt.wavedec2(spectral_images[i,:,:], wavelet)
            haar_images[i,:,:] = self.unwrap_wavelet_coeffs(coeffs)
        return haar_images

    def data_to_spectrogram(self, x_data, window_type = 'hann'):
        f, t, Sxx = spectrogram(x_data, fs=self.sample_rate,
            window=window_type, nperseg=int(self.sample_rate*self.window_len),
            noverlap = int(self.sample_rate*(self.window_len - self.window_lag)))
        # Truncate spectrogram, keep only passband frequencies
        if self.min_freq > 0:
            fidx_keep = (f >= self.min_freq)
            Sxx = Sxx[fidx_keep, :]
            f = f[fidx_keep]
        if self.max_freq < f[-1]:
            fidx_keep = (f <= self.max_freq)
            Sxx = Sxx[fidx_keep, :]
            f = f[fidx_keep]
        self.frequencies = f
        self.times = t
        return f, t, Sxx

    def data_to_haar_images(self, x_data):
        f, t, Sxx = self.data_to_spectrogram(x_data)
        spectral_images, nWindows, idx1, idx2 = self.spectrogram_to_spectral_images(Sxx)
        haar_images = self.spectral_images_to_wavelet(spectral_images)
        haar_images = normalize(self.images_to_vectors(haar_images), axis=1)
        return haar_images, nWindows, idx1, idx2, Sxx, t

    def images_to_vectors(self, images):
        N,d1,d2 = np.shape(images)
        vectors = np.zeros([N,d1*d2])
        for i in range(N):
            vectors[i,:] = np.reshape(images[i,:,:], (1,d1*d2))
        return vectors

    def standardize_haar(self, haar_images):
        haar_means = np.mean(haar_images, axis=0)
        haar_stddevs = np.std(haar_images, axis=0)
        return (haar_images - haar_means)/haar_stddevs

    def binarize_vectors_topK_sign(self, coeff_vectors, k):
        N,M = np.shape(coeff_vectors)
        binary_vectors = np.zeros((N,2*M), dtype=bool)
        for i in range(N):
            idx = np.argsort(abs(coeff_vectors[i,:]))[-k:]
            binary_vectors[i,idx]   = coeff_vectors[i,idx] > 0
            binary_vectors[i,idx+M] = coeff_vectors[i,idx] < 0
        return binary_vectors

    def normalize_and_fingerprint(self, haar_images, fp_file, k=200):
        std_haar_images = self.standardize_haar(haar_images)
        binaryFingerprints = self.binarize_vectors_topK_sign(std_haar_images, k=k)
        # Write to file
        b = np.packbits(binaryFingerprints)
        '''
        with open(fp_file, "a") as out:
            json.dump(b.tolist(), out)
        '''
        return b

    def get_partition_padding(self):
        # add this to end of time series of each partition so we don't have missing fingerprints
        sec_extra = self.spec_len + (self.fp_len - self.fp_lag) * self.spec_lag
        time_extra = datetime.timedelta(seconds=sec_extra)
        return time_extra

    def get_min_fp_length(self):
        return self.fp_len * self.spec_lag + self.spec_len

    def extract_fingerprints(self, filepath, fp_file):
        t_start = time.time()
        # read sac file
        st = read(filepath)
        time_padding = self.get_partition_padding()
        min_fp_length = self.get_min_fp_length()

        fingerprints = []

        for i in range(len(st)):
            # Get start and end time of the current continuous segment
            starttime = datetime.datetime.strptime(str(st[i].stats.starttime), '%Y-%m-%dT%H:%M:%S.%fZ')
            endtime = datetime.datetime.strptime(str(st[i].stats.endtime), '%Y-%m-%dT%H:%M:%S.%fZ')
            # Ignore segments that are shorter than one spectrogram window length
            if endtime - starttime < datetime.timedelta(seconds = min_fp_length):
                continue

            s = starttime
            # Generate and output fingerprints per partition_len
            while endtime - s > datetime.timedelta(seconds = min_fp_length):
                dt = datetime.timedelta(
                    seconds = self.partition_len)
                e = min(s + dt, endtime)
                e_padding = min(s + dt + time_padding, endtime)
                partition_st = st[i].slice(UTCDateTime(s.strftime('%Y-%m-%dT%H:%M:%S.%f')),
                    UTCDateTime(e_padding.strftime('%Y-%m-%dT%H:%M:%S.%f')))
                # Spectrogram + Wavelet transform
                haar_images, nWindows, idx1, idx2, Sxx, t = self.data_to_haar_images(partition_st.data)
                # Write fingerprint time stamps to file
                #write_timestamp(t, idx1, idx2, s, ts_file)
                # Normalize and output fingerprints on a roughly 8 hour time interval
                b = self.normalize_and_fingerprint(haar_images, fp_file)
                fingerprints.append(b)
                s = e
        return fingerprints


