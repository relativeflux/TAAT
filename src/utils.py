import json
import yaml


def dedup(lst):
    ks = sorted(lst)
    return [ks[i] for i in range(len(ks)) if i == 0 or ks[i] != ks[i-1]]

def json_read(filename: str):
    with open(filename) as f:
        return json.load(f)

def json_write(data, filename: str, msg: str):
    with open(filename, "w") as f:
        if msg: print(msg)
        json.dump(data, f, indent=3)

def yaml_read(filename: str):
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def get_dir_size(path):
    size = 0
    for entry in os.scandir(path):
        if entry.is_file():
            size += entry.stat().st_size
        elif entry.is_dir():
            size += get_dir_size(entry.path)
    return size