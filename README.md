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

#### Other dependencies

The Olaf audio fingerprinting library needs to be compiled - find instructions [here](https://github.com/JorenSix/Olaf?tab=readme-ov-file#compilation). You will also need to build the [Olaf Python wrapper](https://github.com/JorenSix/Olaf/blob/master/python-wrapper/README.textile), which creates the Olaf shared C library.

Once you have built these you will need to copy certain resources from the Olaf top-level folder to the top-level folder of the TAAT repository:

- The `bin` folder containing `libolaf.so` (the Olaf shared C library).
- The`olaf_cffi.*.so` shared library (where `*` indicates platform identifier text).

You will also need to set the `LD_LIBRARY_PATH` environment variable (Linux- or Mac-specific command provided below):

`export LD_LIBRARY_PATH=/path/to/TAAT/bin`

### API

**store(_path_)**

Stores extracted fingerprints from an audio file in the database.

Audio is decoded and resampled using ffmpeg, so ffmpeg needs to be available on your system in order for it to work.

| Parameter Name  | Description |
| ------------ | ------------- |
| **path : string**             | Path to the file, or folder of files, whose fingerprints will be extracted and stored in the database.  |

**stats()**

Get statistics on the database.

**query(_path, no_identity_match=True, prune_below=0.2, group_by_path=True_)**

Extracts fingerprints from the supplied audio files and attempts to matche them with what is stored in the database.

| Parameter Name    | Description   |
| ----------------- | ------------- |
| **path : string** | Path to the file to be queried against the database.  |
| **no_identity_match : bool**  | Whether or not to include the queried file in the result, if it is itself already stored in the database.  |
| **prune_below : float>0.0<1.0,scalar>0**  | Matches below this value will be excluded from the result. Accepts either a value between 0.0 and 1.0 (inclusive), in which case the value is interpreted as a percentage of the total match count, or a scalar greater than 0, in which case the value is interpreted as being absolute. |

**write_match_files(_outdir, query_path, matches, sr=22050_)**

Write matches to disk as audio files.

| Parameter Name    | Description   |
| ----------------- | ------------- |
| **outdir : string** | Path to the location to store the folder of audio files created.  |
| **query_path : string**  | Path to a file that has been queried against the database.  |
| **matches : array**  | An array of matches resulting from querying the file supplied via **query_path**. |
| **sr : scalar>0**  | Audio sample rate.  |

### Usage

```python
from olaf_api import *


store("path/to/audio/files/to/store")

query_results = query("path/to/file/to/query.wav")

# Write matches as audio files
write_path_files("path/to/output/folder", "path/to/queried/audio", query_results, sr=44100)
```
