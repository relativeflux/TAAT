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

### Usage

```python
from olaf_api import *


store("path/to/audio/files/to/store")

result = query("path/to/file/to/query.wav")

# Just format the result
get_ref_and_query_values(result)

# Dedupe result
deduped_result = dedupe_matches(result)

# Write matches as audio files
write_path_files("path/to/output/folder", deduped_result, "path/to/ref/audio", "path/to/query/audio", 22050)
```

### API

TBC