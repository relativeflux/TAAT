import os
import random
import numpy as np
import binascii
import librosa
import sklearn
import skimage
import matplotlib.pyplot as plt
from fingerprint_extractor import pickRandomCoeffs, nextPrime, hashcode, jaccard
from waveprint import spectrogram_to_img_data


d1 = set(random.randint(0, 256) for _ in range(1000))
d2 = set(random.randint(0, 256) for _ in range(1000))

a = [6, 6, 6, 6, 6, 5, 5, 5, 5, 5]
b = [5, 5, 5, 5, 5, 6, 6, 6, 6, 6]

def audio_file_to_img_data(filepath, sr=16000, n_fft=1024, hop_length=1024):
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spect = librosa.amplitude_to_db(np.abs(spect), ref=np.max)
    return spectrogram_to_img_data(spect, sr)

def binarize_img(img):
    #img = skimage.io.imread(img_path)
    img = skimage.color.rgb2gray(img[:,:,:3])
    thresh = skimage.filters.threshold_otsu(img)
    return img > thresh

def binarize_img_topK(img, k=200):
    #img = skimage.io.imread(img_path)
    img = skimage.color.rgb2gray(img[:,:,:3])
    n, m = np.shape(img)
    bv = np.zeros((n,m), dtype=bool)
    for i in range(n):
        idx = np.argsort(abs(img[i,:]))[-k:]
        bv[i,idx] = img[i,idx] > 0
    return bv
    
def imshow(img, cmap="gray"):
    plt.imshow(img, cmap=cmap)
    plt.show()

def match_images(filepath1, filepath2, numHashes=256, k=0):
    a = pickRandomCoeffs(numHashes)
    b = pickRandomCoeffs(numHashes)
    img1 = audio_file_to_img_data(filepath1)
    img2 = audio_file_to_img_data(filepath2)
    img1 = binarize_img_topK(img1, k) if k and k>0 else binarize_img(img1)
    img2 = binarize_img_topK(img2, k) if k and k>0 else binarize_img(img2)
    s1 = create_shingleIDset(np.packbits(img1))
    s2 = create_shingleIDset(np.packbits(img2))
    sig1 = get_minhash_signature(s1, numHashes, a, b)
    sig2 = get_minhash_signature(s2, numHashes, a, b)
    jacc = jaccard(sig1, sig2)
    query = test_query(sig1, sig2, numHashes)
    mh = minhash(sig1, sig2, numHashes)
    return f"jaccard: {jacc}, test_query: {query}, minhash: {mh}"

def create_shingleIDset(d):
    shingles = set() #[]
    for i in range(0, len(d) - 2):
        b = f"{d[i]} {d[i+1]} {d[i+2]}".encode("ascii")
        h = binascii.crc32(b) & 0xffffffff
        shingles.add(h)
    return shingles

def minhash(d1, d2, numHashes=256):
    hash_funcs = []
    for i in range(numHashes):
        hash_funcs.append(universal_hashing())
    m1 = [min([h(e) for e in d1]) for h in hash_funcs]
    m2 = [min([h(e) for e in d2]) for h in hash_funcs]
    #minhash_sim = sum(int(m1[i] == m2[i]) for i in range(numHashes)) / numHashes
    minhash_sim = 0
    for k in range(0, numHashes):
        minhash_sim = minhash_sim + (m1[k] == m2[k])
    return minhash_sim / numHashes

def universal_hashing():
    def rand_prime():
        while True:
            p = random.randrange(2 ** 32, 2 ** 34, 2)
            if all(p % n != 0 for n in range(3, int((p ** 0.5) + 1), 2)):
                return p
    m = 2 ** 32 - 1
    p = rand_prime()
    a = random.randint(0, p)
    if a % 2 == 0:
        a += 1
    b = random.randint(0, p)
    def h(x):
        return ((a * x + b) % p) % m
    return h

def get_minhash_signature(shingleIDSet, numHashes, a, b):
    signature = []
    # For each of the random hash functions...
    for i in range(0, numHashes):
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

def test_query(sig1, sig2, numHashes=256):
    count = 0
    for k in range(0, numHashes):
        count = count + (sig1[k] == sig2[k])
    return count / numHashes

##################################################################################
##################################################################################

# Bag of visual words model.

'''
class BOVW():

    def __init__(self, num_cluster=512):
        extractor = skimage.feature.SIFT()
        histograms = []
        kmeans = sklearn.cluster.KMeans(clusters=num_clusters)
        kmeans.fit()

    def get_image_features(self, img_path):
        img = skimage.io.imread(img_path)
        img = skimage.color.rgb2gray(img[:,:,:3])
        self.extractor.detect_and_extract(img)
        return extractor.keypoints, extractor.descriptors

    def build_histogram(self, descriptor_list):
        histogram = np.zeros(len(self.kmeans.cluster_centers_))
        result =  self.kmeans.predict(descriptor_list)
        for i in result:
            histogram[i] += 1.0
        return histogram

    def cluster(self, images):
        for image in images:
        keypoint, descriptor = self.get_image_features(image)
        if (descriptor is not None):
            histogram = self.build_histogram(descriptor)
            self.histograms.append(histogram)
'''

class BOVW():

    def __init__(self, source_dir, n_clusters=64, random_state=None, sr=16000, n_fft=1024, hop_length=512):
        self.source_dir = source_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_clusters = n_clusters
        self.extractor = skimage.feature.SIFT()
        self.docNames = []
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.kmeans = sklearn.cluster.KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )

    def get_spectrogram_descriptors(self, filepath):
        sr=self.sr
        n_fft=self.n_fft
        hop_length=self.hop_length
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        #audio = butter_bandpass_filter(audio, lowcut=180, highcut=3000, fs=sr)
        spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        #spect = librosa.cqt(audio, sr=sr, hop_length=hop_length)
        spect = librosa.amplitude_to_db(np.abs(spect), ref=np.max)
        img_data = spectrogram_to_img_data(spect, sr)
        img_data = skimage.color.rgb2gray(img_data)
        print(f"Extracting descriptors for {os.path.basename(filepath)}")
        self.extractor.detect_and_extract(img_data)
        return self.extractor.descriptors

    # Use this for a folder of images...
    '''
    def get_spectrogram_descriptors(self, filepath):
        img = skimage.io.imread(filepath)
        img = skimage.color.rgb2gray(img[:,:,:3])
        print(f"Extracting descriptors for {os.path.basename(filepath)}")
        self.extractor.detect_and_extract(img)
        return self.extractor.descriptors
    '''

    def descriptors_to_histogram(self, descriptors):
        preds = self.kmeans.predict(descriptors)
        bins = range(self.kmeans.n_clusters)
        return np.histogram(preds, bins=bins, density=True)[0]

    def fit_transform(self):
        descriptors = []
        for dirpath, dirnames, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename.endswith(".wav"):
                    filepath = os.path.join(dirpath, filename)
                    self.docNames.append(os.path.basename(filepath))
                    descr = self.get_spectrogram_descriptors(filepath)
                    descriptors.append(descr)
        print("Fitting descriptors...")
        self.kmeans.fit(np.concatenate(descriptors))
        print("Generating histograms...")
        self.histograms = [self.descriptors_to_histogram(descr) for descr in descriptors]
        result = {}
        for (i, arr) in enumerate(self.histograms):
            result[self.docNames[i]] = list(arr)
        return result

    def query(self, filepath, nn=20):
        descr = self.get_spectrogram_descriptors(filepath)
        hist = self.descriptors_to_histogram(descr)
        NN = sklearn.neighbors.NearestNeighbors(n_neighbors=nn)
        NN.fit(self.histograms)
        dist, result = NN.kneighbors([hist])
        return dist, result
        