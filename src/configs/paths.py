import os
import json

# Get project root dir
conf_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(conf_dir, os.pardir, os.pardir))
data_dir = os.path.join(root, "data")


pathsjson_path = os.path.join(conf_dir, "paths.json")
with open(pathsjson_path, "r") as jin:
    pdict = json.load(jin)

paths = {}
for k, v in pdict["paths"].items():
    paths[k] = os.path.join(root, *v)

