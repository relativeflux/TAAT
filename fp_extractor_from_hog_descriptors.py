import os
import binascii
import numpy as np
from LSH import *
from matplotlib import pyplot as plt
import skimage
import librosa
from hog_descriptor import spectrogram_to_hog_img
from segmentation import get_mfcc_ssm

def sm_to_hog_img_data(S):
    fig, ax = plt.subplots()
    plt.set_cmap("gray")
    plt.imshow(S, interpolation="none")
    plt.xticks([])
    plt.yticks([])
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return data[:, :, :3]

class FPExtractorFromHOGDescriptors:

    def __init__(self, source_dir, analysis_type="melspectrogram", numHashes=10, sr=16000, n_fft=1024, hop_length=512, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), k=200):

        self.source_dir = source_dir
        self.analysis_type = analysis_type
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
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
        orientations = self.orientations
        pixels_per_cell = self.pixels_per_cell
        cells_per_block = self.cells_per_block

        print(f"Processing HOG descriptors for {filepath}")
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        #spect = librosa.cqt(audio, sr=sr, hop_length=hop_length)
        spect = librosa.amplitude_to_db(np.abs(spect), ref=np.max)
        img = spectrogram_to_hog_img(spect, sr)
        #MFCC_S, _ = get_mfcc_ssm(audio)
        #img = sm_to_hog_img_data(MFCC_S)
        fd, _ = skimage.feature.hog(img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            channel_axis=-1,
        )

        '''
        idx = np.argsort(fd)[-self.k:]
        bv = np.zeros([len(fd)], dtype=bool)
        bv[idx] = fd[idx] > 0
        bv = bv.reshape(-1, orientations)
        bv = bv.astype(int).astype(str)
        '''

        fd = fd.reshape(-1, orientations)

        fingerprints = set()

        print(f"Processing fingerprints for {filepath}")
        for i in range(len(fd) - 3):
            '''
            chunk = bv[i:i+2]
            binarized = ["".join(bin) for bin in chunk]
            binarized = [binascii.crc32(bin.encode("ascii")) for bin in binarized]
            a, b = binarized
            fingerprints.add(a | b)
            '''
            chunk = fd[i:i+2]
            [a, b] = chunk
            binarized = [(1 if elt > b[i] else 0) for (i, elt) in enumerate(a)]
            binarized = int(np.packbits(binarized)[0])
            fingerprints.add(binarized)
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
