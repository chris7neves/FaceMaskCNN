from cProfile import label
import os
from collections import Counter

import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.dataset import Dataset


RANDOM_SEED = 42

######################################
#          Masktype Dataset          #
######################################


class MaskTypeDataset(Dataset):
    """
    Pytorch dataset wrapper for the mask type problem.
    Adds convenience methods, and prepares dataset for use with pytorch dataloader.
    """

    def __init__(self, datainfo_df, labels_df, transform=None, grayscale=False, consider_bias=False):
        
        self.labels = labels_df
        self.img_paths = datainfo_df
        self.transform = transform
        self.grayscale = grayscale
        self.consider_bias = consider_bias
        if self.consider_bias:
            self.biases = self.img_paths["im_path"].apply(lambda x: x.split("_")[-1].split(".")[0])

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
        
        if self.consider_bias:
            # The name of the image will contain its bias category
            bias = self.biases.at[idx]
            return image, (label, bias)
        else:
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

def get_masktype_data_df_recursive(paths):
    
    class_dfs = []

    for _class, path in paths.items():
        fpaths = []  
        fnames = []      
        for path, subdirs, files in os.walk(path):
            for name in files:
                fnames.append(name)
                img_path = os.path.join(path, name)
                fpaths.append(img_path)
        df = pd.DataFrame.from_dict({"file_id":fnames, "im_path":fpaths})
        df["label_literal"] = _class
        class_dfs.append(df)
    
    data = pd.concat(class_dfs)
    data = data.reset_index(drop=True)
    data["label_literal"] = pd.Categorical(data["label_literal"])
    data["label"] = data.label_literal.cat.codes
    data.to_csv("FULLDATA.csv")
    return data

def lazy_load_train_val_test(data, labels, train_size, test_size, val_size=0):
    """
    Lazy load the data into train, validation and test sets. Train and test size are specified. 
    """

    if  (1-train_size-test_size-val_size) < 0:
        print("Ratios specified for train/validation/test must sum to 1. Train: {}  Test: {}  Val: {}"
                .format(train_size, test_size, val_size))

    if train_size == 1:
        data_dict = {"train": [data.reset_index(drop=True), labels.reset_index(drop=True)]}
        return data_dict

    train_len = int(train_size * len(labels))
    val_len = int(val_size * len(labels))

    X_other, X_test, y_other, y_test = train_test_split(data, labels,
                                            train_size=(train_len + val_len), 
                                            stratify=labels, 
                                            random_state=RANDOM_SEED)

    data_dict1 = {
        "train": [X_other.reset_index(drop=True), y_other.reset_index(drop=True)],
        "test": [X_test.reset_index(drop=True), y_test.reset_index(drop=True)]
    }
    
    if val_size != 0:
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
    
    if val_size != 0:
        return data_dict2
    else:
        return data_dict1 

def get_masktype_datasets(data_dict, transform = None, grayscale=False, consider_bias=False):
    """
    Prepare the datasets for the masktype problem given the dict containing 
    all the relevant dfs with img_paths and labels.
    """

    # Create the datasets
    datasets = {}
    for t, d in data_dict.items():
        if transform:
            datasets[t] = MaskTypeDataset(d[0], d[1], transform, grayscale=grayscale, consider_bias=consider_bias)
        else:
            datasets[t] = MaskTypeDataset(d[0], d[1], grayscale=grayscale, consider_bias=consider_bias)

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

def get_label_dict(data_df):
    label_dict = dict(list(data_df.groupby(["label_literal", "label"]).indices.keys()))
    label_dict = {v : k for k, v in label_dict.items()}
    return label_dict


def prepare_train_val_strategy(paths_dict, transforms, train_on_everything=False, skip_val=False, search_subdir=False):

    print("Preparing datasets and data loaders ....")
    if search_subdir:
        data_df = get_masktype_data_df_recursive(paths_dict)
    else:
        data_df = get_masktype_data_df(paths_dict)
    label_dict = dict(list(data_df.groupby(["label_literal", "label"]).indices.keys()))
    label_dict = {v : k for k, v in label_dict.items()}
    labels = data_df.pop("label")
    print(label_dict)
    if train_on_everything:
        train_prop = 1.0
        val_prop = 0.0
        test_prop = 0.0
    elif skip_val:
        train_prop = 0.8
        val_prop = 0.0
        test_prop = 1.0 - train_prop
    else:
        train_prop = 0.7
        val_prop = 0.2
        test_prop = 1.0 - val_prop - train_prop

    data_dict = lazy_load_train_val_test(data_df, labels, train_prop, test_prop, val_prop)
    datasets = get_masktype_datasets(data_dict, transforms, grayscale=False)
    dataloaders = get_dataloaders(datasets, train_batch_size=128, val_batch_size=128)
    print("Data is prepared.\n")
    print("Training data has the following distribution:")

    # Print the label distribution in the training set
    train_label_distr = datasets["train"].get_label_distr(label_dict)
    for k, v in train_label_distr.items():
        print("{}: {}".format(k, v))

    # Print the label distribution in the validation set
    if val_prop != 0:
        print("\nValidation data has the following distribution:")
        valid_label_distr = datasets["validation"].get_label_distr(label_dict)
        for k, v in valid_label_distr.items():
            print("{}: {}".format(k, v))

    return dataloaders

def prepare_kfold_strategy(paths_dict, transforms, bias, search_subdir, grayscale=False):

    # Get the data in a df
    if search_subdir:
        data_df = get_masktype_data_df_recursive(paths_dict)
    else:
        data_df = get_masktype_data_df(paths_dict)

    label_dict = dict(list(data_df.groupby(["label_literal", "label"]).indices.keys()))
    label_dict = {v : k for k, v in label_dict.items()}
    print(label_dict)
    labels = data_df.pop("label")
    data_dict = lazy_load_train_val_test(data_df, labels, 1.0, 0.0, 0.0)

    # Create datadict containing only a single set
    if bias:
        datasets = get_masktype_datasets(data_dict, transforms, grayscale=grayscale, consider_bias=True)
    else:
        datasets = get_masktype_datasets(data_dict, transforms, grayscale=grayscale, consider_bias=False)

    label_distr = datasets["train"].get_label_distr(label_dict)
    for k, v in label_distr.items():
        print("{}: {}".format(k, v))
    
    return datasets, label_dict

