# TAAT

## Tape Archive Analysis Toolkit

### Installation

We highly recommend installing into a virtual environment.

#### Install using `venv`

To install using Python's inbuilt `venv` tool `cd` to the TAAT repository and execute the following command:

`python -m venv .venv`

This will create the hidden folder `.venv` at the top level of the repository.

You can call the virtual environment anything, but if you have cloned the repository we recommend only using the names `venv`, `.venv`, `env` or `.env`, as these will be ignored by Git.

To use your virtual environment you will need to activate it, using:

`source .venv/bin/activate`

Note the relative path to the `activate` command. When you are done with the virtual environment you should deactivate it using:

`deactivate`

To install dependencies use the `requirements.txt` file, as follows:

`pip install -r requirements.txt`

#### Install using `conda`

TBC

#### Overview

TBC

### API

**store(_source_dir, features=["melspectrogram"], sr=16000, n_fft=2048, hop_length=1024, k=5, metric="cosine", num_paths=5_)**

Stores extracted cross-similarity data from a folder of audio files in the database.

| Parameter Name  | Description    |
| :-------------- | :------------- |
| **_source_dir : string_**             | Path to the folder of files whose cross-similarity data will be stored in the database.  |
| **_features : list of strings_**             | List of features to extract in the analysis. Available features are: stft, melspectrogram, chroma_cqt, mfcc, rms, tempogram, spectral_centroid, spectral_flatness, spectral_bandwidth and spectral_contrast. |
| **_sr : int>0_**             | Sample rate for the audio loaded for the analysis.  |
| **_n_fft : int>0_**             | FFT analysis frame size.  |
| **_hop_length : int>0_**             | FFT analysis hop length.  |
| **_k : int>0_**             | Number of nearest-neighbours to compute for each analysis sample.  |
| **_metric : string_**             | Distance metric to use for the cross-similarity analysis.  |
| **_num_paths : int>0_**             | Number of RQA paths to compute.  |

**stats(_verbose=False_)**

Get statistics on the database.

| Parameter Name  | Description    |
| :-------------- | :------------- |
| **_verbose : bool_**             | Whether or not to return stats in verbose format.  |

**query(_query_path, no_identity_match=True, verbose=False_)**

Extracts fingerprints from the supplied audio files and attempts to match them with what is stored in the database.

| Parameter Name    | Description   |
| :-----------------| :------------ |
| **_query_path : string_** | Path to the file to be queried against the database.  |
| **_no_identity_match : bool_**  | Whether or not to include the queried file in the result, if it is itself already stored in the database.  |
| **_verbose : bool_** | Whether or not to return the result in verbose format. |

**write(_outdir, query_path, matches, sr=16000_)**

Write matches to disk as audio files.

| Parameter Name     | Description    |
| :----------------- | :------------- |
| **_outdir : string_** | Path to the location to store the folder of audio files created.  |
| **_query_path : string_**  | Path to a file that has been queried against the database.  |
| **_matches : dict_**  | A dictionary of matches resulting from running a query against an audio file. |
| **_sr : int>0_**  | Audio sample rate.  |

### Usage

```python
from taat import store


FEATURES = ["melspectrogram", "tempogram", "rms", "spectral_centroid"]

taat = store(source_dir="path/to/audio/files/to/store",
             features=FEATURES,
             sr=22050,
             k=7,
             n_fft=2048,
             hop_length=1024)

results = taat.query("path/to/file/to/query.wav", verbose=False)

# Write matches as audio files
taat.write("path/to/output/folder", results, sr=22050, format="wav")
```
