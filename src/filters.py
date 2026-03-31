from scipy import signal, ndimage
import numpy as np
import librosa


def butter_bandpass_filter(data, lowcut=180, highcut=3000,
                           sr=16000, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.lfilter(b, a, data)

def median_filter(spect, metric="cosine"):
    return librosa.decompose.nn_filter(spect,
                                       aggregate=np.median,
                                       metric=metric)

def nlm_filter(spect, metric="cosine"):
    rec = librosa.segment.recurrence_matrix(spect,
                                            mode="affinity",
                                            metric=metric,
                                            sparse=True)
    return librosa.decompose.nn_filter(spect,
                                       aggregate=np.average,
                                       rec=rec)

def transient_smoothing_filter(spect, size=[1, 9]):
    return ndimage.median_filter(spect, size=size)

def preemphasis_filter(signal):
    return librosa.effects.preemphasis(signal)