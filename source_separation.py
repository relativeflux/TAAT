import math
import os
import numpy as np
import matplotlib
from matplotlib.pylab import plt
import soundfile as sf


def divergence(V, W, H, beta=2):
    # beta = 2 : Euclidean cost function
    if beta == 0 : return np.sum( V/(W@H) - math.log10(V/(W@H)) -1 )
    # beta = 1 : Kullback-Leibler cost function
    if beta == 1 : return np.sum( V*math.log10(V/(W@H)) + (W@H - V))
    # beta = 0 : Itakura-Saito cost function
    if beta == 2 : return 1/2*np.linalg.norm(W@H-V)


def source_separation(V, S, beta=2, threshold=0.05, max_iter=6000):
    counter = 0
    cost_function = []
    beta_divergence = 1

    K, N = np.shape(V)

    W = np.abs(np.random.normal(loc=0, scale=2.5, size=(K,S)))
    H = np.abs(np.random.normal(loc=0, scale=2.5, size=(S,N)))

    while beta_divergence >= threshold and counter <= max_iter:

        H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
        W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)

        beta_divergence =  divergence(V, W, H, beta = 2)
        cost_function.append(beta_divergence.item())
        counter += 1

    return W, H, cost_function


def plot_cost_function(cost_function):
    plt.plot(range(len(cost_function)), cost_function, label="Cost Function")
    plt.show()

def get_filtered_stfts(W, H, S=2, sr=16000, hop_length=512):
    filtered_stfts = []
    for i in range(S):
        # Filter each source component
        filtered_stft = W[:,[i]]@H[[i],:]
        filtered_stfts.append(filtered_stft)
    return filtered_stfts

def plot_filtered_stfts(filtered_stfts, sr=16000, hop_length=512):
    S = len(filtered_stfts)
    f, axs = plt.subplots(nrows=1, ncols=S, figsize=(20,5))
    for (i, stft) in enumerate(filtered_stfts):
        axs[i].set_title(f"Frequency Mask of Audio Source s = {i+1}")
        # Compute the filtered spectrogram
        D = librosa.amplitude_to_db(stft, ref=np.max)
        # Show the filtered spectrogram
        librosa.display.specshow(D, y_axis="hz", sr=sr, hop_length=hop_length, x_axis="time", cmap=matplotlib.cm.jet, ax=axs[i])
    plt.show()

def reconstruct_audio(filtered_stfts, phase_stft, output_dir, sr=16000, fft_size=1024, hop_length=512):
    for (i, stft) in enumerate(filtered_stfts):
        reconstruct = stft * np.exp(1j * phase_stft)
        layer = librosa.istft(reconstruct, n_fft=fft_size, hop_length=hop_length)
        sf.write(os.path.join(output_dir, f"layer_{i}.wav"), layer, sr)

'''
audio, _ = librosa.load("test5/16000kHz/chunks/005_Disintegration_(op.10)_chunk_3.wav", mono=True, sr=16000)
V = librosa.stft(audio, n_fft=1024, hop_length=512)
phase_stft = np.angle(V)
V = np.abs(V)
V = V + 1e-10

W, H, cost_function = source_separation(V, 2, beta=2, threshold=0.05, max_iter=2000)
'''

def source_separation2(filepath, sr=sr, n_fft=1024, hop_length=1024):
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D_harm, D_perc = librosa.decompose.hpss(D)
    return D, D_harm, D_perc

def plot_source_separation(D, D_harm, D_perc):
    # Pre-compute a global reference power from the input spectrum
    rp = np.max(np.abs(D))
    db_spect = librosa.amplitude_to_db(np.abs(D), ref=rp)
    db_harm = librosa.amplitude_to_db(np.abs(D_harm), ref=rp)
    db_perc = librosa.amplitude_to_db(np.abs(D_perc), ref=rp)
    #
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    #
    img = librosa.display.specshow(db_spect, y_axis='log', x_axis='time', ax=ax[0])
    ax[0].set(title='Full spectrogram')
    ax[0].label_outer()
    #
    librosa.display.specshow(db_harm, y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='Harmonic spectrogram')
    ax[1].label_outer()
    #
    librosa.display.specshow(db_perc, y_axis='log', x_axis='time', ax=ax[2])
    ax[2].set(title='Percussive spectrogram')
    fig.colorbar(img, ax=ax)
    plt.show()