import math
import numpy as np
from matplotlib.pylab import plt


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

'''
plt.plot(range(0,2001), cost_function, label="Cost Function")

f, axs = plt.subplots(nrows=1, ncols=1, figsize=(20,5))
D = librosa.amplitude_to_db(W, ref=np.max)
librosa.display.specshow(D, y_axis="hz", sr=16000, hop_length=512, x_axis="time", cmap=matplotlib.cm.jet, ax=axs)
plt.show()

def filtered_spectrogram(W, H, S=2, sr=16000, hop_length=512):
    f, axs = plt.subplots(nrows=1, ncols=S, figsize=(20,5))
    filtered_spectrograms = []
    for i in range(S):
        axs[i].set_title(f"Frequency Mask of Audio Source s = {i+1}") 
        # Filter each source component
        filtered_spectrogram = W[:,[i]]@H[[i],:]
        # Compute the filtered spectrogram
        D = librosa.amplitude_to_db(filtered_spectrogram, ref=np.max)
        # Show the filtered spectrogram
        librosa.display.specshow(D, y_axis="hz", sr=sr, hop_length=hop_length, x_axis="time", cmap=matplotlib.cm.jet, ax=axs[i])
        filtered_spectrograms.append(filtered_spectrogram)
    #return filtered_spectrograms
    plt.show()

def reconstruct_audio(filtered_spectrograms, n_fft=1024, hop_length=512):
    reconstructed_audio = []
    for i in range(S):
        reconstruct = filtered_spectrograms[i] * np.exp(1j*sound_stft_angle)
        new_sound   = librosa.istft(reconstruct, n_fft=n_fft, hop_length=hop_length)
        reconstructed_audio.append(new_sound)
    return reconstructed_audio
'''