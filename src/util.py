import os
from collections import Counter
from sysconfig import get_path

import pandas as pd
from torch.utils.data.dataset import Dataset
from configs.paths import get_paths


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

def rename_nested_dirs(path_dict):

     for _class, dir in path_dict.items():    
        for path, subdirs, files in os.walk(dir):
            for i, name in enumerate(files):
                _, ext = os.path.splitext(name)
                old_img_path = os.path.join(path, name)
                immediate_parent_folder = old_img_path.split(os.path.sep)[-2]
                new_name = str(i) + "_" + _class[0] + "_" + immediate_parent_folder + ext
                new_img_path = os.path.join(path, new_name)
                try:
                    os.rename(old_img_path, new_img_path)
                except:
                    continue

def get_label_distr(dataset: Dataset, data_dict=None) -> dict:
    """
    Gets the label distribution in a dataset. Iterates through the labels.
    Inspiration: https://stackoverflow.com/questions/62319228/number-of-instances-per-class-in-pytorch-dataset
    """
    
    per_class = [label for _, label in dataset]
    per_class = Counter(per_class)

    return dict(per_class)

def class_dict_from_aug_paths():
    """
    Get the categorical variable, class name pairs from the saved json of class paths.
    """
    class_labels = {i:k for i, (k, v) in enumerate(paths_aug.items())}
    return class_labels

if __name__ == "__main__":

    paths = get_paths("paths_aug_balanced_race")
    rename_nested_dirs(paths)


    pass