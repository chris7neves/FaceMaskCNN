import argparse
from cProfile import label
import os
import sys
from models.fmcnn1 import Fmcnn1
from datasets import get_dataloaders, get_masktype_data_df, get_masktype_datasets, lazy_load_train_val_test

import metrics_and_plotting as mp

import numpy as np

from train import train
from test import test

from configs.paths import paths_aug, paths_cropped, model_dir

import pandas as pd
import seaborn as sns

from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms as T
from torchsummary import summary
import torch.nn.functional as F
# MAKE SURE TO SHUFFLE IMPORT ORDER AND DELETE USELESS IMPORTS
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, Dropout, BatchNorm2d, BCELoss
from torch.optim import Adam, SGD



parser = argparse.ArgumentParser(description="Entrypoint to FaceMaskCNN model.")

subparsers = parser.add_subparsers(dest='mode')

# Training sub parser
train_parser = subparsers.add_parser("train")
train_parser.add_argument("--gen_report", action="store_true")

# Testing sub parser
test_parser = subparsers.add_parser("test")
test_parser. add_argument("from_saved", action="store")
test_parser.add_argument("--gen_report", action="store_true")

# Inference sub parser
infer_parser = subparsers.add_parser("infer")
infer_parser.add_argument("--gen_report", action="store_true")

# List models
list_parser = subparsers.add_parser("list_models")


args = parser.parse_args()

print(args.__str__())  

if args.mode == "train":
    
    # Prepare transform
    masktype_prepr = T.Compose([
        T.ToTensor(),
        T.Resize([64,64]),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #T.Normalize((0.4453, ), (0.2692, ))
        
    ]) 

    # Prepare dataloaders
    data_df = get_masktype_data_df(paths_aug)
    labels = data_df.pop("label")
    data_dict = lazy_load_train_val_test(data_df, labels, 0.7, 0.2, validation=True)
    datasets = get_masktype_datasets(data_dict, masktype_prepr, grayscale=False)
    dataloaders = get_dataloaders(datasets, batch_size=124)

    # Training parameters
    model = Fmcnn1()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    train_losses, validation_losses, validation_accuracies = train(model, dataloaders, 3, optimizer, criterion, validation=True)

    mp.get_train_val_curve(train_losses, validation_losses, validation_accuracies)

    plt.show()

elif args.mode == "test":
     
    savename = args.from_saved

    # Validate savename to make sure it exists
    saved_model_path = os.path.join(model_dir, "saved_models", savename)
    print(saved_model_path)
    if os.path.isfile(saved_model_path):
        model = Fmcnn1()
        model.load_state_dict(torch.load(saved_model_path))
    else:
        print("Invalid .pth or .pt file specified: {}".format(saved_model_path))
        sys.exit(0)

    masktype_prepr = T.Compose([
        T.ToTensor(),
        T.Resize([64,64]),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_df = get_masktype_data_df(paths_aug)

    label_dict = dict(list(data_df.groupby(["label_literal", "label"]).indices.keys()))
    label_dict = {v : k for k, v in label_dict.items()}
    labels = data_df.pop("label")

    data_dict = lazy_load_train_val_test(data_df, labels, 0.7, 0.2, validation=True)
    datasets = get_masktype_datasets(data_dict, masktype_prepr, grayscale=False)
    dataloaders = get_dataloaders(datasets, batch_size=300)
    criterion = CrossEntropyLoss()

    test_labels, test_preds = test(model, dataloaders, criterion)

    f1 = mp.get_f1_score(test_labels, test_preds, average=None)
    print("===========================================================")
    if not isinstance(f1, float):
        for i, f in enumerate(f1):
            
            print("{} F1 Score: {}".format(label_dict[i], f))
    else:
        print("F1 Score: {}".format(f1))
    print("===========================================================")

    cmdf = mp.get_confusion_matrix_df(test_labels, test_preds, label_dict)
    cmax = mp.get_confusion_matrix_ax(cmdf)
    plt.show()


