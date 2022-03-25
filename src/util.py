import os
import numpy as np
from sklearn.decomposition import DictionaryLearning
from torch.utils.data.dataset import Dataset
from collections import Counter

from configs.paths import paths_aug

def rename_images(im_dir, prefix="", suffix=""):
    """
    Utility function that renames all the images in a folder to have increasing integer numbers.
    """
    
    if not os.path.isdir(im_dir):
        print("The specified directory '{}' does not exist.")
        return False
    
    for i, fname in enumerate(os.listdir(im_dir)):
        _, ext = os.path.splitext(fname)
        new_name = prefix + str(i) + suffix + ext
        
        try:
            os.rename(os.path.join(im_dir,fname), os.path.join(im_dir,new_name))
        except FileExistsError:
            continue
            
    return True

def get_label_distr(dataset: Dataset, data_dict=None) -> dict:
    # https://stackoverflow.com/questions/62319228/number-of-instances-per-class-in-pytorch-dataset
    
    per_class = [label for _, label in dataset]
    per_class = Counter(per_class)

    return dict(per_class)

def class_dict_from_aug_paths():
    class_labels = {i:k for i, (k, v) in enumerate(paths_aug.items())}
    return class_labels