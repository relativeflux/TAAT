import os
import math
import mmap
import numpy as np
import librosa
import wave
from tqdm import trange


pitch_shifts = {
    "simple": [12],
    "simple_variable": [4, 7, 12],
    "complex_variable": [2, 4, 5, 7, 9, 11, 12],
    "chromatic_variable": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
}

def pitch_shift_2_time_stretch(pitch_shift=0):
    return 2 ** (pitch_shift / 12)

def write_pitch_shifted_file(input_filepath: str, output_dir: str, sr=16000, chunk_length=10, pitch_shift=0):
    if pitch_shift != 0:
        output_filepath = os.path.join(output_dir, os.path.basename(input_filepath))
        with wave.open(output_filepath, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            time_stretch = pitch_shift_2_time_stretch(pitch_shift)
            for (_, chunk) in stream(input_filepath, chunk_length=chunk_length, overlap=1.0,
                                    print_fn=False, show_progress_bar=False):
                if chunk is not None and len(chunk) != 0:
                    chunk = librosa.effects.pitch_shift(chunk, n_steps=pitch_shift, sr=sr)
                    chunk = librosa.effects.time_stretch(chunk, rate=time_stretch)
                    chunk = (chunk * 32767).astype(np.int16)
                    wav.writeframes(chunk)
        return output_filepath
    else:
        return input_filepath

def walk(dir="", filetype=".wav", only_load_if=lambda filename: filename==filename,
         chunk_length=30, overlap=0.5, print_fn=None, show_progress_bar=True):
    result = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(filetype):
                if callable(only_load_if)==True and only_load_if(filename)==True:
                    filepath = os.path.join(dirpath, filename)
                    for (idx, chunk) in stream(filepath, chunk_length=chunk_length, overlap=overlap, print_fn=print_fn,
                                               show_progress_bar=show_progress_bar):
                        yield((filepath, idx, chunk))

def stream(filepath, sr=16000, chunk_length=30, overlap=0.5, print_fn=None, show_progress_bar=True):
    overlap_secs = int(chunk_length * overlap)
    file_dur = int(librosa.get_duration(path=filepath))
    for (idx, offset) in enumerate(trange(0, math.floor(file_dur-overlap_secs), overlap_secs, disable=(not show_progress_bar))):
        if print_fn and idx==0:
            print(print_fn(filepath))
        chunk, _ = librosa.load(filepath, sr=sr, mono=True, offset=offset, duration=chunk_length)
        yield(idx, chunk)

