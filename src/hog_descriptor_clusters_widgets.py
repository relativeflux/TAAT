import ipywidgets as widgets
from IPython.display import display, Audio, HTML
import numpy as np
import librosa

from hog_descriptor import HOGDescriptorClusters


source_dir_selector = widgets.Text(
    value='test5/16000kHz/chunks7',
    placeholder='Type path to analysis source directory',
    description='Source Dir:',
    disabled=False
)

fft_size_selector = widgets.IntText(
    value=1024,
    description='FFT Size:',
    disabled=False
)

hop_length_selector = widgets.IntText(
    value=512,
    description='Hop Length:',
    disabled=False
)

orientations_selector = widgets.IntText(
    value=8,
    description='Orientations:',
    disabled=False
)

pixels_per_cell_selector = widgets.IntText(
    value=16, #[16,16],
    description='Pixels per cell:',
    disabled=False
)

cells_per_block_selector = widgets.IntText(
    value=1, #[1,1],
    description='Cells per block:',
    disabled=False
)

linkage_selector = widgets.Select(
    options=[
        'ward',
        'single',
        'complete',
        'average',
        'centroid'
    ],
    value='ward',
    # rows=10,
    description='Linkage:',
    disabled=False
)

run_btn = widgets.Button(description="RUN")

output = widgets.Output()

def handle_on_click():
    output.clear_output()
    with output:
        def on_select(filepath):
            audio, _ = librosa.load(filepath, sr=16000, mono=True)
            display(Audio(data=audio, rate=16000))
        hdc = HOGDescriptorClusters(on_select=on_select)
        source_dir = source_dir_selector.value
        n_fft = fft_size_selector.value
        hop_length = hop_length_selector.value
        orientations=orientations_selector.value
        pixels_per_cell=[pixels_per_cell_selector.value, pixels_per_cell_selector.value]
        cells_per_block=[cells_per_block_selector.value, cells_per_block_selector.value]
        hdc.get_clusters(source_dir, n_fft=n_fft, hop_length=hop_length, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        hdc.plot()

run_btn.on_click(lambda btn: handle_on_click())

items = [
    source_dir_selector,
    fft_size_selector,
    hop_length_selector,
    orientations_selector,
    pixels_per_cell_selector,
    cells_per_block_selector,
    run_btn,
    #output
]

controls = widgets.VBox(items)
output_display = widgets.Box([output])

main_widget = widgets.HBox([controls, output_display])

display(main_widget)