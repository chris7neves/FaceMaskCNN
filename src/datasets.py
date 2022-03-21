import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T

from configs.paths import paths, root, data_dir

######################################
#          Masktype Dataset          #
######################################



# import data
# split into test and train

class MaskTypeDataset(Dataset):

    def __init__(self, datainfo_df, labels_df, transform=None):
        self.labels = torch.FloatTensor(labels_df.to_numpy()) 
        self.img_fnames = datainfo_df["im_path"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pass
    #def load_images(self):
        # Only implement if lazy loading takes forever


def get_data_df():

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
                                            random_state=42)
    data_dict1 = {
        "x_train":X_other, "y_train":y_other,
        "x_test":X_test, "y_test":y_test
    }
  
    X_train, X_val, y_train, y_val = train_test_split(X_other, y_other,
                                                    train_size=train_len, 
                                                    test_size=val_len, 
                                                    stratify=y_other, 
                                                    random_state=42)
    data_dict2 = {
        "x_train":X_train, "y_train":y_train,
        "x_test":X_test, "y_test":y_test,
        "x_val":X_val, "y_val":y_val
    }
    
    if validation:
        return data_dict2
    else:
        return data_dict1 

def get_masktype_datasets():

    # Get the labels of the data


    return 


def get_masktype_dataloaders(batch_size=32):


    data_df = get_data_df()
    labels = data_df.pop("label")
    data_dict = lazy_load_train_val_test(data_df, labels, 0.7, 0.2, validation=True)
    


    # dloader = torch.utils.data.DataLoader(dset,
    #                                     batch_size=batch_size,
    #                                     shuffle=True
    #                                     )

    return data_dict


data_dict = get_masktype_dataloaders()
for k, c in data_dict.items():
    print("{}: {}".format(k, len(c)))