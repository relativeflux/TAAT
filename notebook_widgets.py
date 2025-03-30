import ipywidgets as widgets
from IPython.display import display, Audio

from features import FEATURES
from main import run

import librosa
from cross_similarity import get_xsim, plot_xsim, get_xsim_start_end_times
import numpy as np


y_ref_selector = widgets.Text(
    value='',
    placeholder='Type path to y_ref file',
    description='y_ref:',
    disabled=False
)

y_comp_selector = widgets.Text(
    value='',
    placeholder='Type path to y_comp file',
    description='y_comp:',
    disabled=False
)

feature_selector = widgets.Select(
    options=[
        'melspectrogram',
        'mfcc',
        'chroma_cqt',
        'spectral_centroid',
        'spectral_bandwidth',
        'spectral_contrast',
        'spectral_flatness',
    ],
    value='melspectrogram',
    # rows=10,
    description='Feature:',
    disabled=False
)

sample_rate_selector = widgets.IntText(
    value=22050,
    description='Sample Rate:',
    disabled=False
)

fft_size_selector = widgets.IntText(
    value=8192,
    description='FFT Size:',
    disabled=False
)

hop_length_selector = widgets.IntText(
    value=8192,
    description='Hop Length:',
    disabled=False
)

metric_selector = widgets.Select(
    options=[
        'euclidean',
        'cosine',
        'manhattan',
    ],
    value='cosine',
    # rows=10,
    description='Metric:',
    disabled=False
)

k_selector = widgets.IntText(
    value=5,
    description='k:',
    disabled=False
)

gap_onset_selector = widgets.IntText(
    value=5,
    description='Gap Onset:',
    disabled=False
)

gap_extend_selector = widgets.IntText(
    value=10,
    description='Gap Extend:',
    disabled=False
)

knight_moves_selector = widgets.Checkbox(
    value=True,
    description='Knight Moves:',
    disabled=False
)

run_btn = widgets.Button(description="RUN")

output = widgets.Output()

def handle_on_click():
    output.clear_output()
    with output:
        y_ref_path = y_ref_selector.value
        y_comp_path = y_comp_selector.value
        feature = feature_selector.value
        sr = sample_rate_selector.value
        fft_size = fft_size_selector.value
        hop_length = hop_length_selector.value
        metric = metric_selector.value
        k = k_selector.value
        y_ref, _ = librosa.load(y_ref_path, sr=sr, mono=True)
        y_comp, _ = librosa.load(y_comp_path, sr=sr, mono=True)
        a, b = get_xsim(y_comp, y_ref, feature=feature, sr=sr, fft_size=fft_size, hop_length=hop_length, k=k, metric=metric, gap_onset=5, gap_extend=10, knight_moves=True)
        start1, end1, start2, end2 = get_xsim_start_end_times(b, y_ref, y_comp, sr)
        plot_xsim(a, b, start1, end1, start2, end2)
        y_ref_seg, _ = librosa.load(y_ref_path, sr=sr, mono=True, offset=start1, duration=end1-start1)
        display(Audio(data=y_ref_seg, rate=sr))
        y_comp_seg, _ = librosa.load(y_comp_path, sr=sr, mono=True, offset=start2, duration=end2-start2)
        display(Audio(data=y_comp_seg, rate=sr))

run_btn.on_click(lambda btn: handle_on_click())

items = [
    y_ref_selector,
    y_comp_selector,
    feature_selector,
    sample_rate_selector,
    fft_size_selector,
    hop_length_selector,
    metric_selector,
    k_selector,
    run_btn,
]

controls = widgets.VBox(items)
output_display = widgets.Box([output])

main_widget = widgets.HBox([controls, output_display])

display(main_widget)

'''
source_folder_selector = widgets.HTML(
    value="<input type='file' webkitdirectory multiple/>",
    placeholder='Select folder...',
    description='Source Folder:',
)
'''

'''
source_folder_selector = widgets.Text(
    value='../Dropbox/Miscellaneous/TAAT/Data/Test Cases/Test 1/data',
    placeholder='Type path to analysis source folder',
    description='Source Folder:',
    disabled=False
)

chunk_length_selector = widgets.IntText(
    value=8,
    description='Chunk Length:',
    disabled=False
)

features_selector = widgets.SelectMultiple(
    options=FEATURES,
    value=FEATURES,
    description='Features:',
    disabled=False
)

fft_size_selector = widgets.IntText(
    value=2048,
    description='FFT Size:',
    disabled=False
)

hop_length_selector = widgets.IntText(
    value=2048,
    description='Hop Length:',
    disabled=False
)

run_btn = widgets.Button(description="RUN")

output = widgets.Output()

def handle_on_click():
    output.clear_output()
    with output:
        source_folder = source_folder_selector.value
        chunk_length = chunk_length_selector.value
        features = features_selector.value
        fft_size = fft_size_selector.value
        hop_length = hop_length_selector.value
        print(f"Source Folder: {source_folder}\nChunk Length: {chunk_length}\nFeatures: {features}\nFFT Size: {fft_size}\nHop Length: {hop_length}")
        run(source_folder, "./output", chunk_length, features, fft_size, hop_length)

run_btn.on_click(lambda btn: handle_on_click())

items = [
    source_folder_selector,
    chunk_length_selector,
    features_selector,
    fft_size_selector,
    hop_length_selector,
    run_btn,
    #output
]

controls = widgets.VBox(items)
output_display = widgets.Box([output])

main_widget = widgets.HBox([controls, output_display])

display(main_widget)
'''