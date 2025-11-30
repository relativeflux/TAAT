### Installation

We highly recommend installing into a virtual environment.

#### Creating a virtual environment

To create a virtual environment we can use Python's inbuilt `venv` tool. Simply `cd` into the TAAT repository and execute the following command:

`python -m venv .venv`

This will create the hidden folder `.venv` at the top level of the repository.

You can call the virtual environment anything, but if you have cloned the repository we recommend only using the names `venv`, `.venv`, `env` or `.env`, as these will be ignored by Git.

To use your virtual environment you will need to activate it, using:

`source .venv/bin/activate`

Note the relative path to the `activate` command. When you are done with the virtual environment you should deactivate it using:

`deactivate`

#### Install using `pip`

To install taat using `pip`, from the top-level TAAT directory execute the following:

`python -m pip install .`

#### Install using `conda`

TBC

### Basic Usage

The code below provides an example of basic TAAT usage. For more advanced examples see the [Tutorials](tutorials.md) page.

```python
from taat import query


FEATURES = ["melspectrogram", "tempogram", "rms", "spectral_centroid"]

# Run a query.
query_result = query(source_dir="path/to/audio/files/to/query/against",
                     query_filepath="path/to/file/to/query.wav",
                     features=FEATURES,
                     sr=16000,
                     k=7,
                     n_fft=2048,
                     hop_length=1024)

# Write matches to disk as audio files
query_result.write("path/to/output/folder")
```

### The taat_interpreter Max patch

We also have a Max patch available, the taat_interpreter (created by Dr Sam Gillies), which allows to load exported JSON output from a query result and display the it as playable audio waveforms:

![image](img/taat_interpreter.png)
