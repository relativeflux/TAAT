import os
import math
import binascii
import numpy as np
from matplotlib import pyplot as plt
import skimage
import librosa
from cross_similarity import get_xsim
from fingerprint_extractor import hashcode, pickRandomCoeffs, maxShingleID, nextPrime

# Adapted from https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d/65547931#65547931
def trim_value(arr, value=255):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only the value provided.
    """
    zeroed = np.where(arr < value, arr, 0)
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(zeroed))
    return arr[slices]

def get_xsim_img_data(xsim, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(xsim, sr=sr, ax=ax)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return data[:, :, :3]

def create_shingleIDset(d):
    shingles = set() #[]
    for i in range(0, len(d) - 3):
        b = f"{d[i]} {d[i+1]} {d[i+2]}".encode("ascii")
        h = binascii.crc32(b) & 0xffffffff
        shingles.add(h)
    return shingles

class XSIMFingerprintExtractor():

    def __init__(self, source_dir, numHashes=64, sr=16000, n_fft=8192, hop_length=8192, k=20, metric="cosine", mode="affinity"):
        self.source_dir = source_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.k = k
        self.metric = metric
        self.mode = mode
        self.docNames = []
        self.numHashes = numHashes
        self.fingerprints = {}
        self.signatures = []
        self.coeffA = pickRandomCoeffs(self.numHashes)
        self.coeffB = pickRandomCoeffs(self.numHashes)

    def get_xsim_fingerprints(self, filepath1, filepath2):
        sr = self.sr
        n_fft = self.n_fft
        hop_length = self.hop_length
        k = self.k
        metric = self.metric
        mode = self.mode
        samples1, _ = librosa.load(filepath1, sr=sr, mono=True)
        samples2, _ = librosa.load(filepath2, sr=sr, mono=True)
        xsim, _ = get_xsim(samples1, samples2, feature="melspectrogram", fft_size=n_fft, hop_length=hop_length, k=k, metric=metric, mode=mode)
        xsim_data = get_xsim_img_data(xsim, sr)
        skimage.io.imsave(f"./imgs/{os.path.basename(filepath1)}.png", xsim_data)
        #xsim_data = trim_value(xsim_data, value=255)
        img = skimage.color.rgb2gray(xsim_data)
        thresh = skimage.filters.threshold_otsu(img)
        img = img > thresh
        return create_shingleIDset(np.packbits(img))

    def extract_fingerprints(self):
        for dirpath, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename.endswith(".wav"):
                    filepath = os.path.join(dirpath, filename)
                    self.docNames.append(filepath)
                    fingerprints = self.get_xsim_fingerprints(filepath, filepath)
                    self.fingerprints[filepath] = fingerprints

    def get_minhash_signature(self, shingleIDSet, a, b):
        signature = []
  
        # For each of the random hash functions...
        for i in range(0, self.numHashes):
            
            minHashCode = nextPrime + 1

            # For each shingle in the document...
            for (j, shingleID) in enumerate(shingleIDSet):
                h = hashcode(shingleID, a[i], b[i], nextPrime)
                
                # Track the lowest hash code seen...
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
        a = self.coeffA
        b = self.coeffB

        matches = {}

        for i in range(0, len(self.docNames)):
            ref_sig = self.signatures[i]
            fingerprints = self.get_xsim_fingerprints(filepath, self.docNames[i])
            query_sig = self.get_minhash_signature(fingerprints, a, b)
            count = 0
            # Count the number of positions in the signatures which are equal...
            for k in range(0, self.numHashes):
                if len(ref_sig) >= self.numHashes:
                    count = count + (ref_sig[k] == query_sig[k])
                res = count / self.numHashes
            if res > 0:
                matches[self.docNames[i]] = res

        return matches

                    
                    
        