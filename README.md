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