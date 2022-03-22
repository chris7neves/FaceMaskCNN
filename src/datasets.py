import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T

from configs.paths import root, data_dir

######################################
#          Masktype Dataset          #
######################################


class MaskTypeDataset(Dataset):

    def __init__(self, datainfo_df, labels_df, transform=None):
        
        self.labels = labels_df
        self.img_paths = datainfo_df
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #self.img_paths.to_csv("impathsdebug.csv")
        im_path = self.img_paths.at[idx, "im_path"]
        image = imread(im_path, as_gray=False)
        label = self.labels.at[idx]
        if self.transform:
            image = self.transform(image)

        # if(image.size(dim=0) == 4):
        #     image = image[:3, :, :]
        #     temp = torch.swapaxes(image, 0, 1)
        #     temp = torch.swapaxes(temp, 1, 2)
        #     imshow(temp)
        #     plt.show()

        return image, label

    def get_path(self, idx):
        return self.img_paths.at[idx, "im_path"]
        


    #def load_images(self):
        # Only implement if lazy loading takes forever


def get_masktype_data_df(paths):

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
        "train": [X_other.reset_index(drop=True), y_other.reset_index(drop=True)],
        "test": [X_test.reset_index(drop=True), y_test.reset_index(drop=True)]
    }
  
    X_train, X_val, y_train, y_val = train_test_split(X_other, y_other,
                                                    train_size=train_len, 
                                                    test_size=val_len, 
                                                    stratify=y_other, 
                                                    random_state=42)
    data_dict2 = {
        "train": [X_train.reset_index(drop=True), y_train.reset_index(drop=True)],
        "test": [X_test.reset_index(drop=True), y_test.reset_index(drop=True)],
        "validation": [X_val.reset_index(drop=True), y_val.reset_index(drop=True)]
    }
    
    if validation:
        return data_dict2
    else:
        return data_dict1 

def get_masktype_datasets(data_dict, transform = None):

    # Create the datasets
    datasets = {}
    for t, d in data_dict.items():
        if transform:
            datasets[t] = MaskTypeDataset(d[0], d[1], transform)
        else:
            datasets[t] = MaskTypeDataset(d[0], d[1])

    return datasets


def get_dataloaders(datasets, batch_size=32):

    dataloaders = {}
    for t, s in datasets.items():
        dataloaders[t] = torch.utils.data.DataLoader(s, batch_size=batch_size, shuffle=True)
       
    return dataloaders




# masktype_prepr = T.Compose([
#     T.ToTensor(),
#     T.Resize([32,32])
# ])
    
# data_df = get_data_df()
# labels = data_df.pop("label")
# data_dict = lazy_load_train_val_test(data_df, labels, 0.7, 0.2, validation=True)
# datasets = get_masktype_datasets(data_dict, masktype_prepr)
# dataloaders = get_dataloaders(datasets)

# feature, label = next(iter(dataloaders["validation"]))
# print(f"Feature batch shape: {feature.size()}")
# print(f"Labels batch shape: {label.size()}")
# img = feature[0].squeeze()
# label = label[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")