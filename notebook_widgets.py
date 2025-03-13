import ipywidgets as widgets
from IPython.display import display

from features import FEATURES
from main import run


'''
source_folder_selector = widgets.HTML(
    value="<input type='file' webkitdirectory multiple/>",
    placeholder='Select folder...',
    description='Source Folder:',
)
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