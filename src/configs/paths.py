import os
import json

# Project root directory
conf_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(conf_dir, os.pardir, os.pardir))

# Project data directory
data_dir = os.path.join(root, "data")

# Project model directory
model_dir = os.path.join(root, "src", "models")

# Project generated reports directory
report_dir = os.path.join(root, "reports")

# Get the paths to all the data folders found in paths.json
pathsjson_path = os.path.join(conf_dir, "paths.json")
with open(pathsjson_path, "r") as jin:
    pdict = json.load(jin)

# Generate all of the paths to the augmented image folders
paths_aug = {}
for k, v in pdict["paths_aug"].items():
    paths_aug[k] = os.path.join(root, *v)

# The paths to all of the image folders containing cropped raw images
paths_cropped = {}
for k, v in pdict["paths_cropped"].items():
    paths_cropped[k] = os.path.join(root, *v)
