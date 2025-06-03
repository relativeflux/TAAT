import os
import librosa
import soundfile as sf
import contextlib
from olaf import Olaf, OlafCommand
from olaf_cffi import lib


def store(path):
    if os.path.isfile(path):
        Olaf(OlafCommand.STORE,path).do()
    elif os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    filepath = os.path.join(dirpath, filename)
                    with contextlib.redirect_stdout(None):
                        Olaf(OlafCommand.STORE,filepath).do()
                    print(f'Stored {filepath}.')

def stats():
    config = lib.olaf_config_default()
    db = lib.olaf_db_new(config.dbFolder,True)
    lib.olaf_db_stats(db, False)
    lib.olaf_db_destroy(db)
    lib.olaf_config_destroy(config)

def query(path, no_identity_match=True, prune_below=0.2):
    with contextlib.redirect_stdout(None):
        result = Olaf(OlafCommand.QUERY,path).do()
    max_match_count = result[0]['matchCount']
    def filter_fn(x):
        match_count = x['matchCount']
        if prune_below < 1: match_count = match_count / max_match_count
        return match_count >= prune_below
    result = list(filter(filter_fn, result))
    if no_identity_match:
        def filter_fn(x):
            x_path = str(x['path'], encoding='utf-8')
            return os.path.basename(path) != os.path.basename(x_path)
        result = list(filter(filter_fn, result))
    return result

def group_matches_by_path(matches):
    paths = []
    result = []
    for match in matches:
        if match['path'] not in paths:
            paths.append(match['path'])
            result.append([match])
        else:
            path_idx = paths.index(match['path'])
            result[path_idx].append(match)
    return result

def get_ref_and_query_values(matches):
    matches = [[[match['queryStart'], match['queryStop']], \
        [match['referenceStart'], match['referenceStop']]] \
            for match in matches]
    return sorted(matches)

def dedupe_matches(matches, margin=2):
    matches = get_ref_and_query_values(matches)
    result = [matches[0]]
    for match in matches[1:]:
        if match[0][1] > result[-1][0][1]:
            #if abs(match[0][0] - result[-1][0][1]) > margin:
            if abs(result[-1][0][1] - match[0][0]) <= margin:
                result.append(match)
    return result

def dedupe_matches2(matches, margin=2):
    matches = get_ref_and_query_values(matches)
    result = [matches[0]]
    for match in matches[1:]:
        if match[0][1] > result[-1][0][1] or \
                match[1][1] > result[-1][1][1]:
            if abs(result[-1][0][1] - match[0][0]) <= margin or \
                    abs(result[-1][1][1] - match[1][0]) <= margin:
                result[-1] = [ [result[-1][0][0], match[0][1]], \
                    [result[-1][1][0], match[1][1]] ]
    return result

def write_path_file(outdir, file_path, filename_prefix, start, end, sr):
    seg, _ = librosa.load(file_path, sr=sr, mono=True, offset=start, duration=end-start)
    filename = f"{filename_prefix}.start={start}_end={end}.wav"
    sf.write(os.path.join(outdir, filename), seg, sr)

# matches should be the output from dedupe_matches.
# y_ref_path and y_comp_path are the paths to the matched and queried files, respectively.
def write_path_files(outdir, matches, y_ref_path, y_comp_path, sr):
    y_ref, _ = librosa.load(y_ref_path, sr=sr, mono=True)
    y_comp, _ = librosa.load(y_comp_path, sr=sr, mono=True)
    for (i, start_end_pair) in enumerate(matches):
        [[start1, end1], [start2, end2]] = start_end_pair
        write_path_file(outdir, y_ref_path, f"y_ref_{i}", start1, end1, sr)
        write_path_file(outdir, y_comp_path, f"y_comp_{i}", start2, end2, sr)

'''
results = [31.288002014160156, 54.35200119018555, 27.784000396728516, 54.10400390625, 32.88800048828125, 54.0160026550293, 105.26400756835938]

pprint.pp(dedupe_matches(results))

[[[27.784000396728516, 31.840002059936523],
  [12.40000057220459, 16.45600128173828]],
 [[31.288002014160156, 34.512001037597656],
  [15.840001106262207, 19.064001083374023]],
 [[54.0160026550293, 80.6719970703125],
  [41.76000213623047, 68.41600036621094]],
 [[105.26400756835938, 110.23200988769531],
  [89.21600341796875, 94.18400573730469]]]
'''