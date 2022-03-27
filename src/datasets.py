import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T

from configs.paths import root, data_dir

RANDOM_SEED = 42

######################################
#          Masktype Dataset          #
######################################


class MaskTypeDataset(Dataset):
    """
    Pytorch dataset wrapper for the mask type problem.
    Adds convenience methods, and prepares dataset for use with pytorch dataloader.
    """

    def __init__(self, datainfo_df, labels_df, transform=None, grayscale=False):
        
        self.labels = labels_df
        self.img_paths = datainfo_df
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        """Overridden len function. Returns len of dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Overridden getitem method. Returns an image and its label. Also applies transform to image."""

        im_path = self.img_paths.at[idx, "im_path"]
        image = imread(im_path, as_gray=self.grayscale)
    
        label = self.labels.at[idx]

        if (not self.grayscale) and (image.shape[2] == 4):
            image = image[:, :, :3]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def get_path(self, idx):
        """Get the path of an image in the dataset."""
        return self.img_paths.at[idx, "im_path"]

    def get_label_distr(self, class_dict):
        """
        Get the number of images per class in the dataset. 
        Class dict is a dictionary that link an into to the corresponding class.
        This is usually generated using the path order in paths.json.
        """
        d = dict(Counter(self.labels.to_list()))
        distr =  {class_dict[k] : v for (k, v) in d.items()}
        return distr
        
def get_masktype_data_df(paths):
    """
    Read the img paths into a df with corresponding label literal and categorical label.
    Used for enabling lazy loading of images.
    """

    class_dfs = []
    
    for _class, path in paths.items():
        df = pd.DataFrame(os.listdir(path)).rename({0:"file_id"}, axis=1)
        df["im_path"] = df["file_id"].apply(lambda id: os.path.join(path, id))
        df["label_literal"] = _class
        class_dfs.append(df)
    
    data = pd.concat(class_dfs)
    data = data.reset_index(drop=True)
    data["label_literal"] = pd.Categorical(data["label_literal"])
    data["label"] = data.label_literal.cat.codes

    return data

def lazy_load_train_val_test(data, labels, train_size, test_size, validation=True):
    """
    Lazy load the data into train, validation and test sets. Train and test size are specified. If
    validation is true, then the remainder from 1 - train_size - test_size = val_size.
    """

    if validation:
        if (1-train_size-test_size) <= 0:
            print("Validation is true but set sizes do not allow for it. Train: {}  Test: {}"
                    .format(train_size, test_size))
        else:
            val_size = 1-train_size-test_size
            val_len = int(val_size * len(labels))

    train_len = int(train_size * len(labels))
    test_len = int(test_size * len(labels))

    X_other, X_test, y_other, y_test = train_test_split(data, labels,
                                            train_size=train_len+val_len, 
                                            test_size=test_len, 
                                            stratify=labels, 
                                            random_state=RANDOM_SEED)

    data_dict1 = {
        "train": [X_other.reset_index(drop=True), y_other.reset_index(drop=True)],
        "test": [X_test.reset_index(drop=True), y_test.reset_index(drop=True)]
    }
  
    X_train, X_val, y_train, y_val = train_test_split(X_other, y_other,
                                                    train_size=train_len, 
                                                    test_size=val_len, 
                                                    stratify=y_other, 
                                                    random_state=RANDOM_SEED)
    data_dict2 = {
        "train": [X_train.reset_index(drop=True), y_train.reset_index(drop=True)],
        "test": [X_test.reset_index(drop=True), y_test.reset_index(drop=True)],
        "validation": [X_val.reset_index(drop=True), y_val.reset_index(drop=True)]
    }
    
    if validation:
        return data_dict2
    else:
        return data_dict1 

def get_masktype_datasets(data_dict, transform = None, grayscale=False):
    """
    Prepare the datasets for the masktype problem given the dict containing 
    all the relevant dfs with img_paths and labels.
    """

    # Create the datasets
    datasets = {}
    for t, d in data_dict.items():
        if transform:
            datasets[t] = MaskTypeDataset(d[0], d[1], transform, grayscale=grayscale)
        else:
            datasets[t] = MaskTypeDataset(d[0], d[1], grayscale=grayscale)

    return datasets


def get_dataloaders(datasets, train_batch_size=32, val_batch_size=32, test_batch_size=32):
    """
    Given the datasets, prepare and return the dataloaders in a dict. 
    "test", "train" and "validation" are the corresponding keys for the dataloaders in the returned dictionary.
    """

    dataloaders = {}
    for t, s in datasets.items():
        if t == "train":
            batch_size = train_batch_size
        elif t == "test":
            batch_size = test_batch_size
        elif t == "validation":
            batch_size = val_batch_size

        dataloaders[t] = torch.utils.data.DataLoader(s, batch_size=batch_size, shuffle=True)
       
    return dataloaders
