import os
import json

# Get project root dir
conf_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(conf_dir, os.pardir, os.pardir))
data_dir = os.path.join(root, "data")
model_dir = os.path.join(root, "src", "models")
report_dir = os.path.join(root, "reports")

pathsjson_path = os.path.join(conf_dir, "paths.json")
with open(pathsjson_path, "r") as jin:
    pdict = json.load(jin)

paths_aug = {}
for k, v in pdict["paths_aug"].items():
    paths_aug[k] = os.path.join(root, *v)

paths_cropped = {}
for k, v in pdict["paths_cropped"].items():
    paths_cropped[k] = os.path.join(root, *v)
