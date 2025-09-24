import os
import json
import binascii
import numpy as np
import sqlite3
from peak_extraction import *
from LSH import *
import librosa
import soundfile as sf
from matplotlib import pyplot as plt


# Adapted from Olaf/src/olaf_fp_extractor.c

class FingerprintExtractorOLD:

    def __init__(self, db_path=".taat/taat.db"):

        self.db_path = db_path

        db_folder = os.path.dirname(self.db_path)

        if not os.path.exists(db_folder):
            os.makedirs(db_folder)

        self.db = sqlite3.connect(self.db_path)

        self.db.cursor().execute("CREATE TABLE fingerprints(path, hash)")

        self.max_event_point_usages = 10
        self.audio_block_index = 0

        self.min_time_distance = 2
        self.max_time_distance = 33
        self.min_freq_distance = 1
        self.max_freq_distance = 128

        self.fingerprints = []
        self.max_fingerprints = 300


    def hash(self, fp):
        f1 = fp['bin1']
        f2 = fp['bin2']
        f3 = fp['bin3'] or 0
        
        t1 = fp['time1']
        t2 = fp['time2']
        t3 = fp['time3'] or 0
        
        m1 = fp['magnitude1']
        m2 = fp['magnitude2']
        m3 = fp['magnitude3'] or 0
        
        f1LargerThanF2 = 1 if f2 > f3 else 0
        f2LargerThanF3 = 1 if f2 > f3 else 0
        f3LargerThanF1 = 1 if f3 > f1 else 0
        
        m1LargerThanm2 = 1 if m1 > m2 else 0
        m2LargerThanm3 = 1 if m2 > m3 else 0
        m3LargerThanm1 = 1 if m3 > m1 else 0
        
        m1LargerThanm2 = 0
        m2LargerThanm3 = 0
        m3LargerThanm1 = 0
        
        dt1t2LargerThant3t2 = 1 if (t2 - t1) > (t3 - t2) else 0
        df1f2LargerThanf3f2 = 1 if abs(f2 - f1) > abs(f3 - f2) else 0

	    # 9 bits f in range( 0 - 512) to 8 bits
        f1Range = (f1 >> 1)

	    # 7 bits (0-128) -> 5 bits
        df2f1 = (abs(f2 - f1) >> 2)
        df3f2 = (abs(f3 - f2) >> 2)

	    # 6 bits max
        diffT = (t3 - t1)
        
        hash = ((diffT                &  ((1<<6)  -1)   ) << 0 ) + \
            ((f1LargerThanF2       &  ((1<<1 ) -1)   ) << 6 ) + \
            ((f2LargerThanF3       &  ((1<<1 ) -1)   ) << 7 ) + \
            ((f3LargerThanF1       &  ((1<<1 ) -1)   ) << 8 ) + \
            ((m1LargerThanm2       &  ((1<<1 ) -1)   ) << 9 ) + \
            ((m2LargerThanm3       &  ((1<<1 ) -1)   ) << 10) + \
            ((m3LargerThanm1       &  ((1<<1 ) -1)   ) << 11) + \
            ((dt1t2LargerThant3t2  &  ((1<<1 ) -1)   ) << 12) + \
            ((df1f2LargerThanf3f2  &  ((1<<1 ) -1)   ) << 13) + \
            ((f1Range              &  ((1<<8 ) -1)   ) << 14) + \
            ((df2f1                &  ((1<<6 ) -1)   ) << 22) + \
            ((df3f2                &  ((1<<6 ) -1)   ) << 28)
            
        return hash


    def extract_fingerprints(self):

        peaks = self.peaks
        peaks = sorted(peaks, key=lambda x: x['time'])

        for (i, peak) in enumerate(peaks):

            t1 = peaks[i]['time']
            f1 = peaks[i]['bin']
            m1 = peaks[i]['magnitude']
            u1 = peaks[i]['usages']

            # Do not evaluate empty points.
            if (f1==0 and t1==0) : break

            # Do not reuse each event point too much.
            if (u1 > self.max_event_point_usages) : break

            # Do not evaluate event points that are too recent.
            diff_to_current_time = self.audio_block_index-self.max_time_distance
            #if (t1 > diff_to_current_time) : break

            for j in range(i+1, len(peaks)):
                
                t2 = peaks[j]['time']
                f2 = peaks[j]['bin']
                m2 = peaks[j]['magnitude']
                u2 = peaks[j]['usages']

                f_diff = abs(f1 - f2)
                t_diff = t2 - t1

                assert(t2>=t1)
                assert(t_diff>=0)

                # Do not reuse each event point too much.
                if (u2 > self.max_event_point_usages) : break

                # Do not evaluate points to far in the future.
                if (t_diff > self.max_time_distance) : break

                if (t_diff >= self.min_time_distance and t_diff <= self.max_time_distance and
                    f_diff >= self.min_freq_distance and f_diff <= self.max_freq_distance):

                    assert(t2>t1)

                    if (len(self.fingerprints) == self.max_fingerprints):
                        print(f'Warning: Fingerprint maximum amount reached, fingerprints are ignored, consider increasing max_fingerprints if you see this often.')
                    else:
                        self.fingerprints.append({
                            'time1': t1,
                            'time2': t2,
                            'time3': t2,
                            'bin1': f1,
                            'bin2': f2,
                            'bin3': f2,
                            'magnitude1': m1,
                            'magnitude2': m2,
                            'magnitude3': m2,
                        })

                        # Count event point usages.
                        self.peaks[i]['usages'] += 1
                        self.peaks[j]['usages'] += 1

                assert(len(self.fingerprints) <= self.max_fingerprints)

        return self.fingerprints


    def store_single(self, filepath, sr=16000, n_fft=2048, hop_length=1024, threshold=2.75):
        _, peaks = find_peaks(filepath, sr, n_fft, hop_length, threshold)
        self.peaks = [
            {'time': t, 'bin': f, 'magnitude': m, 'usages': 0} for (t, f, m) in peaks
        ]
        self.current_path = filepath
        self.extract_fingerprints()
        sql_entries = []
        for fingerprint in self.fingerprints:
            hash = self.hash(fingerprint)
            sql_entries.append(f"('{filepath}', {hash}),")
        sql_entries = "\n    " + "\n    ".join(sql_entries)[:-1]
        self.db.cursor().execute(f"INSERT INTO fingerprints VALUES {sql_entries}")
        self.db.commit()


    def store(self, input):
        if os.path.isfile(input):
            self.store_single(input)
        elif os.path.isdir(input):
            for dirpath, dirnames, filenames in os.walk(input):
                for filename in filenames:
                    if filename.endswith(".wav"):
                        filepath = os.path.join(dirpath, filename)
                        self.store_single(filepath)

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

'''
Sample test data from:
https://medium.com/@hbrylkowski/locality-sensitive-hashing-explained-304eb39291e4

A = ['Who', 'was', 'the', 'first', 'king', 'of', 'Poland']
B = ['Who', 'was', 'the', 'first', 'ruler', 'of', 'Poland']
C = ['Who', 'was', 'the', 'last', 'pharaoh', 'of', 'Egypt']

[["last", "Who", "Egypt", "king", "ruler", "was", "of", "Poland", "pharaoh", "the", "first"],
["the", "of", "Poland", "was", "first", "ruler", "Who", "Egypt", "pharaoh", "last", "king"],
["first", "king", "Egypt", "was", "Who", "of", "pharaoh", "last", "Poland", "ruler", "the"],
["ruler", "king", "Poland", "Who", "the", "pharaoh", "of", "first", "Egypt", "last", "was"],
["king", "Poland", "ruler", "last", "pharaoh", "the", "Who", "Egypt", "first", "of", "was"],
["the", "pharaoh", "Who", "ruler", "Poland", "Egypt", "king", "last", "was", "first", "of"]]
'''


def jaccard(a, b):
    a = set(a)
    b = set(b)
    intersection = a.intersection(b)
    union = a.union(b)
    return len(intersection) / len(union)

def minhash(docs, n=5):
    union = list(set().union(*docs))
    rows = []
    for i in range(0, n):
        perm = np.random.permutation(union)
        for (j, elt) in enumerate(perm, start=1):
            row = list(np.zeros(len(docs), dtype=int))
            for (k, doc) in enumerate(docs):
                if elt in doc : row[k] = j
            rows.append(row)
    out = []
    new_shape = [n,len(union),len(docs)]
    for elt in np.reshape(rows, new_shape):
        s = np.transpose(elt)
        t = []
        for u in s:
            v = next(x for x in u if x>0)
            t.append(v)
        out.append(t)
    return np.transpose(out)

def hashcode(x, a, b, c):
    return (a*x + b) % c

maxShingleID = 2**32-1
nextPrime = 4294967311

def pickRandomCoeffs(k):
  # Create a list of 'k' random values.
  randList = []
  
  while k > 0:
    # Get a random shingle ID.
    randIndex = np.random.randint(0, maxShingleID)
  
    # Ensure that each random number is unique.
    while randIndex in randList:
      randIndex = np.random.randint(0, maxShingleID)
    
    # Add the random number to the list.
    randList.append(randIndex)
    k = k - 1
    
  return randList

class FingerprintExtractorORIGINAL:

    def __init__(self, source_dir, analysis_type="melspectrogram", numHashes=10, sr=16000, n_fft=1024, hop_length=1024, peak_threshold=2.75):

        self.source_dir = source_dir
        self.analysis_type = analysis_type
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.peak_threshold = peak_threshold

        self.numHashes = numHashes

        self.docNames = []

        self.fingerprints = {}
        self.signatures = []

        self.coeffA = pickRandomCoeffs(self.numHashes)
        self.coeffB = pickRandomCoeffs(self.numHashes)

    '''
    def extract_fingerprints_for_file(self, filepath):
        with open(filepath) as f:
            words = f.readline().split(" ")
            shinglesInDoc = set()
            for i in range(0, len(words) - 2):
                shingle = (words[i] + " " + words[i + 1] + " " + words[i + 2]).encode("ascii")
                crc = binascii.crc32(shingle) & 0xffffffff
                shinglesInDoc.add(crc)
        return shinglesInDoc
    '''

    def extract_fingerprints_for_file(self, filepath):

        analysis_type = self.analysis_type
        sr = self.sr
        n_fft = self.n_fft
        hop_length = self.hop_length
        threshold = self.peak_threshold

        _, peaks = find_peaks(filepath, analysis_type, sr, n_fft, hop_length, threshold)
        peaks = sorted(peaks)

        fingerprints = set()

        for i in range(len(peaks) - 3):
            chunk = peaks[i:i+2]
            [[t1, f1, m1], [t2, f2, m2]] = chunk

            '''
            d = { 't1': t1, 't2': t2, 'f1': f1, 'f2': f2, 'm1': m1, 'm2': m2 }
            d = json.dumps(d).encode("ascii")
            '''
            
            #d = f"{t1} {f1} {m1} {t2} {f2} {m2}".encode("ascii")
            #d = binascii.crc32(d) & 0xffffffff

            d = 1 + f2 + f1 * (2 ** 8) + abs(t2-t1) * (2 ** 16)

            fingerprints.add(d)
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

#################################################################################
#################################################################################
#################################################################################

class FingerprintExtractor:

    def __init__(self, source_dir, analysis_type="melspectrogram", numHashes=10, sr=16000, n_fft=1024, hop_length=1024, peak_threshold=2.75, time_interval_max=5):

        self.source_dir = source_dir
        self.analysis_type = analysis_type
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.peak_threshold = peak_threshold

        self.numHashes = numHashes
        self.time_interval_max = time_interval_max

        self.docNames = []

        self.fingerprints = {}
        self.signatures = []

        self.coeffA = pickRandomCoeffs(self.numHashes)
        self.coeffB = pickRandomCoeffs(self.numHashes)

    '''
    def extract_fingerprints_for_file(self, filepath):
        with open(filepath) as f:
            words = f.readline().split(" ")
            shinglesInDoc = set()
            for i in range(0, len(words) - 2):
                shingle = (words[i] + " " + words[i + 1] + " " + words[i + 2]).encode("ascii")
                crc = binascii.crc32(shingle) & 0xffffffff
                shinglesInDoc.add(crc)
        return shinglesInDoc
    '''

    def extract_fingerprints_for_file(self, filepath):

        analysis_type = self.analysis_type
        sr = self.sr
        n_fft = self.n_fft
        hop_length = self.hop_length
        threshold = self.peak_threshold

        _, peaks = find_peaks(filepath, analysis_type, sr, n_fft, hop_length, threshold)
        peaks = sorted(peaks)

        fingerprints = set()

        '''
        for i in range(len(peaks) - 3):
            chunk = peaks[i:i+2]
            [[t1, f1, m1], [t2, f2, m2]] = chunk

            #d = { 't1': t1, 't2': t2, 'f1': f1, 'f2': f2, 'm1': m1, 'm2': m2 }
            #d = json.dumps(d).encode("ascii")
            
            #d = f"{t1} {f1} {m1} {t2} {f2} {m2}".encode("ascii")
            #d = binascii.crc32(d) & 0xffffffff

            d = 1 + f2 + f1 * (2 ** 8) + abs(t2-t1) * (2 ** 16)

            fingerprints.add(d)
        '''

        for i, anchor in enumerate(peaks):
            for target in peaks[i + 1:]:
                if target[0] > anchor[0] and target[0] - anchor[0] < self.time_interval_max:
                    h = (anchor[1] << 22) | (target[1] << 12) | (target[0] - anchor[0])
                    fingerprints.add((h, anchor[0], target[0]))
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
            fp_idx = 0

            # For each shingle in the document...
            for (j, shingleID) in enumerate(shingleIDSet):
                h = hashcode(shingleID[0], a[i], b[i], nextPrime)
                
                # Track the lowest hash code seen.
                if h < minHashCode:
                    minHashCode = h
                    fp_idx = j

            # Add the smallest hash code value as component number 'i' of the signature.
            signature.append((minHashCode, fp_idx))
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

    def query(self, filepath, verbose=True, no_identity_match=True):
        fingerprints = self.extract_fingerprints_for_file(filepath)

        a = self.coeffA
        b = self.coeffB

        query_sig = self.get_minhash_signature(fingerprints, a, b)

        matches = {} #[]

        '''
        for i in range(0, len(self.docNames)):
            ref_sig = self.signatures[i]
            count = 0
            # Count the number of positions in the signatures which are equal...
            for k in range(0, self.numHashes):
                if len(ref_sig) >= self.numHashes:
                    count = count + (ref_sig[k][0] == query_sig[k][0])
                res = count / self.numHashes
            if res > 0:
                matches[self.docNames[i]] = res
        '''
        '''
        for i in range(0, len(self.docNames)):
            ref_sig = self.signatures[i]
            count = 0
            # Count the number of positions in the signatures which are equal...
            for k in range(0, self.numHashes):
                if len(ref_sig) >= self.numHashes:
                    file = self.docNames[i]
                    is_match = (ref_sig[k][0] == query_sig[k][0])
                    if is_match and os.path.basename(file) != os.path.basename(filepath):
                        fp = list(self.fingerprints[file])
                        matchID, ref_start, ref_stop = self.get_fingerprint_info(fp, ref_sig[k])
                        _, query_start, query_stop = self.get_fingerprint_info(list(fingerprints), query_sig[k])
                        matches.append({
                            'matchCount': 0,
                            'path': file,
                            'matchIdentifier': matchID, #ref_sig[k][0], #fp[hash_idx][0]
                            'queryStart': query_start,
                            'queryStop': query_stop,
                            'referenceStart': ref_start,
                            'referenceStop': ref_stop
                        })
                    count = count + is_match
        '''
        for i in range(0, len(self.docNames)):
            ref_sig = self.signatures[i]
            count = 0
            # Count the number of positions in the signatures which are equal...
            for k in range(0, self.numHashes):
                if len(ref_sig) >= self.numHashes:
                    file = self.docNames[i]
                    is_match = (ref_sig[k][0] == query_sig[k][0])
                    if is_match and no_identity_match==True and os.path.basename(file) != os.path.basename(filepath):
                        fp = list(self.fingerprints[file])
                        matchID, ref_start, ref_stop = self.get_fingerprint_info(fp, ref_sig[k], n_fft=self.n_fft, hop_length=self.n_fft)
                        _, query_start, query_stop = self.get_fingerprint_info(list(fingerprints), query_sig[k], n_fft=self.n_fft, hop_length=self.n_fft)
                        match = {
                            #'path': file,
                            'matchIdentifier': matchID, #ref_sig[k][0], #fp[hash_idx][0]
                            'queryStart': query_start,
                            'queryStop': query_stop,
                            'referenceStart': ref_start,
                            'referenceStop': ref_stop
                        }
                        if file not in matches:
                            matches[file] = [match]
                        else:
                            matches[file].append(match)

        if verbose:
            return matches
        else:
            return self.parse_query_output(filepath, matches)

    def get_fingerprint_info(self, fingerprint, signature, n_fft, hop_length):
        [id, idx] = signature
        start_time = fingerprint[idx][1]
        start_time = librosa.frames_to_time(start_time, n_fft=n_fft, hop_length=hop_length)
        stop_time = fingerprint[idx][2]
        stop_time = librosa.frames_to_time(stop_time, n_fft=n_fft, hop_length=hop_length)
        return id, float(start_time), float(stop_time)

    '''
    def parse_query_output(self, query_output):
        result = {}
        keys = list(query_output.keys())
        for (i, v) in enumerate(list(query_output.values())):
            k = keys[i]
            result[f"file{i}"] = k
            segments = []
            for match in v:
                segments.append([match["queryStart"],match["queryStop"]])
                segments.append([match["referenceStart"],match["referenceStop"]])
            result[f"file{i}-segments"] = segments
        return result
    '''

    def parse_query_output(self, query_file_path, query_output):
        result = {}
        keys = list(query_output.keys())
        for (i, v) in enumerate(list(query_output.values())):
            k = keys[i]
            result[f"results_{i}"] = {
                "query_file": os.path.basename(query_file_path),
                "query_segments": [[match["queryStart"], match["queryStop"]] for match in v],
                "reference_file": os.path.basename(k),
                "reference_segments": [[match["referenceStart"], match["referenceStop"]] for match in v]
            }
            
            
        return result

    def lsh(self, b):
        lsh = LSH(b)
        for signature in self.signatures:
            lsh.add_hash(signature)
        candidate_pairs = lsh.check_candidates()
        return [[self.docNames[a], self.docNames[b]] \
            for (a,b) in candidate_pairs]

def write_match_file(outdir, file_path, filename_prefix, start, end, sr):
    seg, _ = librosa.load(file_path, sr=sr, mono=True, offset=start, duration=end-start)
    filename = f"{filename_prefix}.start={start}_end={end}.wav"
    sf.write(os.path.join(outdir, filename), seg, sr)

def make_dir_for_file_path(parent_dir, file_path):
    file_path_dir = os.path.splitext(os.path.basename(file_path))[0]
    file_path_dir = ''.join(['_' if char == ' ' else char for char in file_path_dir])
    file_path_dir = os.path.join(parent_dir, file_path_dir)
    if not os.path.exists(file_path_dir):
        os.makedirs(file_path_dir)
    return file_path_dir

def write_match_files(outdir, query_path, matches, sr=22050):
    matches_dir = make_dir_for_file_path(outdir, query_path)
    keys = list(matches.keys())
    for (idx, match_group) in enumerate(matches.values()):
        #match_path = str(match_group[0]['path'], encoding="utf-8")
        match_path = keys[idx]
        match_group_dir = make_dir_for_file_path(matches_dir, match_path)
        for (i, match) in enumerate(match_group):
            write_match_file(match_group_dir, query_path, f"query_{i}", match["queryStart"], match["queryStop"], sr)
            write_match_file(match_group_dir, match_path, f"ref_{i}", match["referenceStart"], match["referenceStop"], sr)


