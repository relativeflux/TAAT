# Adapted from Olaf/src/olaf_fp_extractor.c

class FingerprintExtractor:

    def __init__(self, peaks):
        self.peaks = [
            {'time': t, 'bin': f, 'magnitude': m, 'usages': 0} for (t, f, m) in peaks
        ]

        self.max_event_point_usages = 10
        self.audio_block_index = 0

        self.min_time_distance = 2
        self.max_time_distance = 33
        self.min_freq_distance = 1
        self.max_freq_distance = 128

        self.fingerprints = []
        self.fingerprint_index = 0
        self.max_fingerprints = 300


    def extract(self):

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
