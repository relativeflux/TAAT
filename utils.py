import json

def dedup(lst):
    ks = sorted(lst)
    return [ks[i] for i in range(len(ks)) if i == 0 or ks[i] != ks[i-1]]

def json_read(filename: str):
    with open(filename) as f:
        return json.load(f)
