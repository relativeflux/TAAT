import pywt
from PIL import Image, ImageOps


def get_wavelet(filepath):
    img = Image.open(filepath)
    img = img.convert("RGB")
    rgb = img.split()
    [[r,_],[g,_],[b,_]] = [pywt.dwt2(chan, "haar") for chan in rgb]
    rgb = [Image.fromarray(arr) for arr in [r,g,b]]
    return Image.merge("RGB", rgb)
