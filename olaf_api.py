import os
import librosa
import soundfile as sf
from olaf import Olaf, OlafCommand


def store(path):
    if os.path.isfile(path):
        Olaf(OlafCommand.STORE,path).do()
    elif os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    filepath = os.path.join(dirpath, filename)
                    Olaf(OlafCommand.STORE,filepath).do()

def query(path):
    return Olaf(OlafCommand.QUERY,path).do()

def dedupe_matches(matches, margin=2):
    matches = [[[match['queryStart'], match['queryStop']], \
        [match['referenceStart'], match['referenceStop']]] \
            for match in matches]
    matches = sorted(matches)
    result = [matches[0]]
    for match in matches:
        if match[0][1] > result[-1][0][1]:
            if abs(match[0][0] - result[-1][0][0]) > margin:
                result.append(match)
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