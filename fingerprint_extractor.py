import os
import sqlite3
from peak_extraction import find_peaks


# Adapted from Olaf/src/olaf_fp_extractor.c

class FingerprintExtractor:

    def __init__(self, dp_path):

        self.db_path = db_path or ".taat/taat.db"

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        self.db = sqlite3.connect(self.db_path)

        self.db.cursor().execute("CREATE TABLE fingerprints(path, hash)")

        self.max_event_point_usages = 10
        self.audio_block_index = 0

        self.min_time_distance = 2
        self.max_time_distance = 33
        self.min_freq_distance = 1
        self.max_freq_distance = 128

        self.fingerprints = []
        self.fingerprint_index = 0
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

	    hash = 
            ((diffT                &  ((1<<6)  -1)   ) << 0 ) +
            ((f1LargerThanF2       &  ((1<<1 ) -1)   ) << 6 ) +
            ((f2LargerThanF3       &  ((1<<1 ) -1)   ) << 7 ) +
            ((f3LargerThanF1       &  ((1<<1 ) -1)   ) << 8 ) +
            ((m1LargerThanm2       &  ((1<<1 ) -1)   ) << 9 ) +
            ((m2LargerThanm3       &  ((1<<1 ) -1)   ) << 10) +
            ((m3LargerThanm1       &  ((1<<1 ) -1)   ) << 11) +
            ((dt1t2LargerThant3t2  &  ((1<<1 ) -1)   ) << 12) +
            ((df1f2LargerThanf3f2  &  ((1<<1 ) -1)   ) << 13) +
            ((f1Range              &  ((1<<8 ) -1)   ) << 14) +
            ((df2f1                &  ((1<<6 ) -1)   ) << 22) +
            ((df3f2                &  ((1<<6 ) -1)   ) << 28)
	
	return hash


    def extract_fingerprints(self):

        peaks = self.peaks

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
            if (t1 > diff_to_current_time) : break

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

                    if (self.fingerprint_index == self.max_fingerprints):
                        print(f'Warning: Fingerprint maximum index {self.fingerprint_index} reached, fingerprints are ignored, consider increasing max_fingerprints if you see this often.')
                    else:
                        self.fingerprints[self.fingerprint_index] = {
                            'time1': t1,
                            'time2': t2,
                            'bin1': f1,
                            'bin2': f2,
                            'magnitude1': m1,
                            'magnitude2': m2,
                        }

                        # Count event point usages.
                        self.peaks[i]['usages'] += 1
                        self.peaks[j]['usages'] += 1

                        self.fingerprint_index += 1

                assert(self.fingerprint_index <= self.max_fingerprints)

        return self.fingerprints


    def store_single(self, filepath, sr=16000, n_fft=2048, hop_length=1024, threshold=2.75):
        peaks = find_peaks(filepath, sr, n_fft, hop_length, threshold)
        self.peaks = [
            {'time': t, 'bin': f, 'magnitude': m, 'usages': 0} for (t, f, m) in peaks
        ]
        self.current_path = filepath
        self.extract_fingerprints()
        sql_entries = []
        for fingerprint in self.fingerprints:
            hash = self.hash(fingerprint)
            sql_entries.append(f"('{filepath}', {hash}),")
        sql_entries = "\n    " + "\n    ".join(sql_entries)
        self.db.cursor().execute(f"INSERT INTO fingerprints VALUES {sql_entries}")
        self.db.commit()


    def store(self, input):
        if os.path.isfile(path):
            self.store_single(input)
        elif os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(".wav"):
                        filepath = os.path.join(dirpath, filename)
                        self.store_single(filepath)

