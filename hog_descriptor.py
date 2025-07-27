import skimage
import matplotlib.pyplot as plt


def get_hog_descriptor(filepath, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1)):
    img = skimage.io.imread(filepath)
    fd, hog = skimage.feature.hog(img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
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