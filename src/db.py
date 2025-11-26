import io
import os
import pathlib
import sqlite3
import json
import numpy as np


# Adapted from https://pythonforthelab.com/blog/storing-data-with-sqlite 

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def connect2db(source_dir, db_dir=None):
    home = pathlib.Path.home()
    db_dir = db_dir or os.path.join(home, ".taat")
    os.makedirs(db_dir, exist_ok=True)
    db = sqlite3.connect(os.path.join(db_dir, "db"), detect_types=sqlite3.PARSE_DECLTYPES)
    table_name = os.path.basename(source_dir)
    db.execute(f"CREATE TABLE IF NOT EXISTS {table_name}(filename VARCHAR, xsim array, rqa array, paths VARCHAR)")
    db.commit()
    return db

def list_table_cols(db, table_name):
    cursor = db.execute(f"select * from {table_name}")
    return list(map(lambda x: x[0], cursor.description))

def insert_row_data(db, table_name, filepath, xsim, rqa, paths):
    paths_list = [p.tolist() for p in paths]
    return db.execute(f"INSERT INTO {table_name} (filename, xsim, rqa, paths) VALUES(?,?,?,?)", [os.path.basename(filepath), xsim, rqa, json.dumps(paths_list)])

def get_row_data(db, table_name, filepath):
    cursor = db.execute(f"SELECT xsim, rqa, paths FROM {table_name} WHERE filename = '{os.path.basename(filepath)}'")
    xsim, rqa, paths = cursor.fetchone()
    return xsim, rqa, json.loads(paths)

'''
from db import *
from cross_similarity import *


filepath1 = "data/test5/16kHz/chunks/001_End_of_the_World_(op.1)_chunk_1.wav"

xsim, rqa, paths, _ = get_xsim_multi(filepath1, filepath1, features=['melspectrogram', 'rms', 'spectral_centroid'], fft_size=2048, hop_length=1024, k=5, metric='cosine', gap_onset=5, gap_extend=10, num_paths=10, enhance=True)

db = connect2db("data/test5/16kHz/chunks")

insert_row_data(db, "chunks", filepath1, xsim, rqa, paths)
row_data = get_row_data(db, "chunks", filepath1)
'''