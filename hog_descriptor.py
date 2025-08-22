import os
import numpy as np
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
import skimage
import matplotlib.pyplot as plt
from matplotlib.text import Text
import librosa


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
    n = len(fd_list)
    # create an empty nxn distance matrix
    distance_matrix = np.zeros((n,n))
    for i in range(n):
        fd_i = fd_list[i]
        for k in range(i):
            fd_k = fd_list[k]
            # measure Jensen–Shannon distance between each feature vector
            # and add to the distance matrix
            distance_matrix[i, k] = distance.jensenshannon(fd_i, fd_k)
    # symmetrize the matrix as distance matrix is symmetric
    return np.maximum(distance_matrix, distance_matrix.transpose())

def spectrogram_to_hog_img(spect, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spect, y_axis='linear', x_axis='time', sr=sr, ax=ax)
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return data[:, :, :3]

def get_hog_descriptor_clusters(source_dir, sr=16000, n_fft=1024, hop_length=512, metric='ward', orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1)):
    fd_list = []
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                filepath = os.path.join(dirpath, filename)
                #print(f"Processing HOG descriptors for {filepath}")
                audio, _ = librosa.load(filepath, sr=sr, mono=True)
                spect = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                spect = librosa.amplitude_to_db(np.abs(spect), ref=np.max)
                img = spectrogram_to_hog_img(spect, sr)
                fd, _ = skimage.feature.hog(img,
                    orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    visualize=True,
                    channel_axis=-1,
                )
                fd_list.append(fd)
    distance_matrix = get_distance_matrix(fd_list)
    cond_distance_matrix = distance.squareform(distance_matrix)
    Z = linkage(cond_distance_matrix, method='ward', metric=metric)
    return Z

def plot_hog_descriptor_clusters(Z, source_dir, figsize=(12,6)):
    files = []
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                files.append(filename)
    fig = plt.figure(figsize=figsize, picker=True)
    def onpick(event):
        for obj in fig.findobj(Text):
            if obj.contains(event.mouseevent)[0]:
                filename = obj.get_text()
                filepath = os.path.join(dirpath, filename)
                print(filepath)
    def onhover(event):
        for obj in fig.findobj(Text):
            if obj.contains(event)[0]:
                obj.set_color("red")
                fig.canvas.draw_idle()
    dendrogram(Z, color_threshold=0.2, show_leaf_counts=True, orientation="left", leaf_label_func=lambda id: files[id])
    fig.canvas.mpl_connect("pick_event", onpick)
    fig.canvas.mpl_connect("motion_notify_event", onhover)
    plt.tight_layout()
    plt.show()

class HOGDescriptorClusters():

    def __init__(self, on_select=lambda x: print(x)):
        self.source_dir = None
        self.files = []
        self.current_selection = None
        self.fig = None
        self.Z = None
        self.on_select = on_select

    def get_clusters(self, source_dir, sr=16000, n_fft=1024, hop_length=512, metric='ward', orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1)):
        self.source_dir = source_dir
        self.Z = get_hog_descriptor_clusters(source_dir=source_dir, sr=sr, n_fft=n_fft, hop_length=hop_length, metric=metric, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

    def set_text_color(self, text, color):
        text.set_color(color)
        self.fig.canvas.draw_idle()

    def onpick(self, event):
        #for obj in self.fig.findobj(Text):
        for obj in self.fig_objects:
            if isinstance(obj, Text):
                if obj.contains(event.mouseevent)[0]:
                    filename = obj.get_text()
                    filepath = os.path.join(self.source_dir, filename)
                    self.on_select(filepath)

    def onhover(self, event):
        #for obj in self.fig.findobj(Text):
        for obj in self.fig_objects:
            if isinstance(obj, Text):
                if obj.contains(event)[0]:
                    if not self.current_selection:
                        self.current_selection = obj
                        self.set_text_color(self.current_selection, "red")
                    else:
                        if self.current_selection.get_text() != obj.get_text():
                            self.set_text_color(self.current_selection, "black")
                            self.current_selection = obj
                            self.set_text_color(self.current_selection, "red")
            '''
            else:
                if self.current_selection:
                    self.set_text_color(self.current_selection, "black")
                    self.current_selection = None
            '''

    def plot(self, figsize=(12, 6)):
        if len(self.Z) > 0:
            for dirpath, dirnames, filenames in os.walk(self.source_dir):
                for filename in filenames:
                    if filename.endswith(".wav"):
                        self.files.append(filename)
            self.fig = plt.figure(figsize=figsize, picker=True)
            dendrogram(self.Z, color_threshold=0.2, show_leaf_counts=True, orientation="left", leaf_label_func=lambda id: self.files[id])
            self.fig.canvas.mpl_connect("pick_event", self.onpick)
            self.fig.canvas.mpl_connect("motion_notify_event", self.onhover)
            #self.text_objects = self.fig.findobj(Text)
            self.fig_objects = self.fig.findobj()
            plt.tight_layout()
            plt.show()

'''
from hog_descriptor import *
hdc = HOGDescriptorClusters()
hdc.get_clusters(source_dir="test5/16000kHz/chunks7", metric="cosine")
hdc.plot()
'''