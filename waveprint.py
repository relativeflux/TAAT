import os
import math
import binascii
import numpy as np
from LSH import *
from matplotlib import pyplot as plt
import skimage
import librosa
import pywt
from sklearn.preprocessing import normalize
from fingerprint_extractor import hashcode, pickRandomCoeffs, maxShingleID, nextPrime
from segmentation import get_mfcc_ssm
from cross_similarity import butter_bandpass_filter


def spectrogram_to_img_data(spect, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spect, sr=sr, ax=ax)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return data[:, :, :3]

def ssm_to_img_data(S):
    fig, ax = plt.subplots()
    plt.set_cmap("gray")
    plt.imshow(S, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return data[:, :, :3]

def unwrap_wavelet_coeffs(coeffs):
        L = len(coeffs)
        cA = coeffs[0]
        for i in range(1,L):
            (cH, cV, cD) = coeffs[i]
            cA = np.concatenate((np.concatenate((cA, cV),axis= 1),np.concatenate((cH, cD),axis = 1)),axis=0)
        return cA

def images_to_vectors(images):
    N,d1,d2 = np.shape(images)
    vectors = np.zeros([N,d1*d2])
    for i in range(N):
        vectors[i,:] = np.reshape(images[i,:,:], (1,d1*d2))
    return vectors

def standardize_haar(haar_images):
    np.seterr(divide='ignore', invalid='ignore')
    haar_means = np.mean(haar_images, axis=0)
    haar_stddevs = np.std(haar_images, axis=0)
    haar_images = (haar_images - haar_means)/haar_stddevs
    for arr in haar_images:
        for (i, elt) in enumerate(arr):
            if math.isnan(elt):
                arr[i] = 0
    return haar_images

def binarize_vectors_topK_sign(coeff_vectors, K):
    N,M = np.shape(coeff_vectors)
    binary_vectors = np.zeros((N,2*M), dtype=bool)
    for i in range(N):
        idx = np.argsort(abs(coeff_vectors[i,:]))[-K:]
        binary_vectors[i,idx] = coeff_vectors[i,idx] > 0
        binary_vectors[i,idx+M] = coeff_vectors[i,idx] < 0
    return binary_vectors

class Waveprint:

    def __init__(self, source_dir, analysis_type="melspectrogram", numHashes=10, sr=16000, n_fft=1024, hop_length=512, k=200):

        self.source_dir = source_dir
        self.analysis_type = analysis_type
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.k = k

        self.numHashes = numHashes

        self.docNames = []

        self.fingerprints = {}
        self.signatures = []

        self.coeffA = pickRandomCoeffs(self.numHashes)
        self.coeffB = pickRandomCoeffs(self.numHashes)

    def extract_fingerprints_for_file(self, filepath):

        analysis_type = self.analysis_type
        sr = self.sr
        n_fft = self.n_fft
        hop_length = self.hop_length

        fingerprints = set()

        print(f"Computing Wavelet representation for {filepath}")
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        audio = butter_bandpass_filter(audio, lowcut=180, highcut=3000, fs=sr)
        spect = False
        if analysis_type=="stft":
            spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            spect = np.abs(spect)
        elif analysis_type=="melspectrogram":
            spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        else:
            spect = librosa.cqt(audio, sr=sr, hop_length=hop_length)
        spect = librosa.amplitude_to_db(np.abs(spect), ref=np.max)
        img_data = spectrogram_to_img_data(spect, sr)
        #MFCC_S, _ = get_mfcc_ssm(audio)
        #img_data = ssm_to_img_data(MFCC_S)
        coeffs = pywt.wavedec2(img_data, pywt.Wavelet("db1"))
        coeffs = unwrap_wavelet_coeffs(coeffs)
        haar_images = normalize(images_to_vectors(coeffs), axis=1)
        haar_images = standardize_haar(haar_images)
        bv = binarize_vectors_topK_sign(haar_images, self.k)
        b = np.packbits(bv)
        print(f"Processing fingerprints for {filepath}")
        for i in range(0, len(b) - 3):
            d = f"{b[i]} {b[i+1]} {b[i+2]}".encode("ascii")
            h = binascii.crc32(d) & 0xffffffff
            fingerprints.add(h)

        return fingerprints

    def extract_fingerprints(self):
        for dirpath, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename.endswith(".wav"):
                    filepath = os.path.join(dirpath, filename)
                    self.docNames.append(filepath)
                    fingerprints = self.extract_fingerprints_for_file(filepath)
                    self.fingerprints[filepath] = fingerprints

    def get_minhash_signature(self, shingleIDSet, a, b):
        signature = []
  
        # For each of the random hash functions...
        for i in range(0, self.numHashes):
            
            minHashCode = nextPrime + 1
            
            # For each shingle in the document...
            for shingleID in shingleIDSet:
                h = hashcode(shingleID, a[i], b[i], nextPrime)
                
                # Track the lowest hash code seen.
                if h < minHashCode:
                    minHashCode = h

            # Add the smallest hash code value as component number 'i' of the signature.
            signature.append(minHashCode)
        return signature

    def generate_minhash_sigs(self):

        a = self.coeffA
        b = self.coeffB
        
        for docID in self.docNames:

            # Get the shingle set for this document.
            shingleIDSet = self.fingerprints[docID]
            
            signature = self.get_minhash_signature(shingleIDSet, a, b)
            
            # Store the MinHash signature for this document.
            self.signatures.append(signature)

    def store(self):
        self.extract_fingerprints()
        self.generate_minhash_sigs()

    def query(self, filepath):
        fingerprints = self.extract_fingerprints_for_file(filepath)

        a = self.coeffA
        b = self.coeffB

        query_sig = self.get_minhash_signature(fingerprints, a, b)

        matches = {}

        for i in range(0, len(self.docNames)):
            ref_sig = self.signatures[i]
            count = 0
            # Count the number of positions in the signatures which are equal...
            for k in range(0, self.numHashes):
                if len(ref_sig) >= self.numHashes:
                    count = count + (ref_sig[k] == query_sig[k])
                res = count / self.numHashes
            if res > 0:
                matches[self.docNames[i]] = res

        return matches


    def lsh(self, b):
        lsh = LSH(b)
        for signature in self.signatures:
            lsh.add_hash(signature)
        candidate_pairs = lsh.check_candidates()
        return [[self.docNames[a], self.docNames[b]] \
            for (a,b) in candidate_pairs]
