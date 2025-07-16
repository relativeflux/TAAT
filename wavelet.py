import pywt
from skimage import io, color


def get_wavelet(filepath):
    img = io.imread(filepath)
    gs = color.rgb2gray(img)
    coeffs = pywt.dwt2(gs, "haar")
    return coeffs