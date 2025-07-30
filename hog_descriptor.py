import scipy
import skimage
import matplotlib.pyplot as plt


def get_hog_descriptor(filepath, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1)):
    img = skimage.io.imread(filepath)
    fd, hog = skimage.feature.hog(img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        channel_axis=-1,
    )
    return img, fd, hog

def plot_hog_descriptor(orig_img, hog_img):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(orig_img, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    # Rescale histogram for better display
    hog_img_rescaled = skimage.exposure.rescale_intensity(hog_img, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_img_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

def get_distance_matrix(fd_list):
    # create an empty nxn distance matrix
    distance_matrix = np.zeros(fd_list.shape)
    for i in range(fd_list.shape[0]):
        fd_i = fd_list[i]
        for k in range(i):
            fd_k = fd_list[k]
            # measure Jensen–Shannon distance between each feature vector
            # and add to the distance matrix
            distance_matrix[i, k] = distance.jensenshannon(fd_i, fd_k)
    # symmetrize the matrix as distance matrix is symmetric
    return np.maximum(distance_matrix, distance_matrix.transpose())

'''
def spectrogram_to_hog_img(spect, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spect, y_axis='linear', x_axis='time', sr=sr, ax=ax)
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba())
    return data[:, :, 3]

def get_hog_descriptor_clusters(source_dir, sr=16000, n_fft=1024, hop_length=512):
    fd_list = []
    for dirpath, dirnames, filenames in os.walk(self.source_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                filepath = os.path.join(dirpath, filename)
                audio, _ = librosa.load(filepath, sr=sr, mono=True)
                spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                spect = np.abs(spect)
                img = spectrogram_to_hog_img(spect)
                _, fd, _ = get_hog_descriptor(img, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1))
                fd_list.append(fd)
    distance_matrix = get_distance_matrix(fd_list)
    cond_distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
    Z = scipy.cluster.hierarchy.linkage(cond_distance_matrix, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, color_threshold=0.2, show_leaf_counts=True)
plt.show()
'''
