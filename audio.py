import librosa


def load(path):
    sr = librosa.get_samplerate(path)
    (audio, _) = librosa.load(path, sr=sr)
    return (path, sr, audio)

def stream(path, block_length=256, frame_length=2048, hop_length=2048, *args, **kwargs):
    sr = librosa.get_samplerate(path)
    stream = librosa.stream(path,
                            block_length=block_length,
                            frame_length=frame_length,
                            hop_length=hop_length)
    return (path, sr, stream)

def load_audio(path):
    if librosa.get_duration(path=path) <= 60:
        return load(path)
    else:
        return stream(path)
